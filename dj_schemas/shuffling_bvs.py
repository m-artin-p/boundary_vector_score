import datajoint as dj
import numpy as np 
import random
import opexebo
from tqdm.auto import tqdm


# Load base schema
schema = dj.schema(dj.config['dj_imaging.database'])
schema.spawn_missing_classes()

# Load personal schema 
borderscore_schema = dj.schema('user_martipof_im')
# Schema components
from dj_schemas.bvs import BVFieldParams, BVScoreParams, BVScoreFieldMethod

from bvs.detect_fields import detect_fields
from bvs.bv_score import calc_bv_score

from helpers.utils import find_nearest, calc_signal_dict, calc_ratemap


@borderscore_schema
class ShuffledBVS(dj.Computed):
    definition = """
    # Shuffling table for boundary vector score (BVS)
    -> FilteredEvents.proj(signal_dataset = 'dataset_name')
    -> Tracking.OpenField.proj(tracking_dataset = 'dataset_name')
    -> ShuffleParams
    -> SignalTrackingParams
    -> MapParams
    -> BVFieldParams
    -> BVScoreParams
    -> BVScoreFieldMethod
    ---
    -> [nullable] Sync.proj(sync_dataset_frames_imaging = 'dataset_name', sync_name_frames_imaging  = 'sync_name')
    number_shuffles            :  int            # Total number of shuffles (can vary from expected number)
    shuffling_offsets          :  blob@imaging_store  # Shuffling offsets
    """

    class BVS(dj.Part):
        definition = """
        # Shuffled bvs  
        -> master
        --- 
        bvs_99                 :  double          #  BVS 99th percentile
        bvs_95                 :  double          #  BVS 95th percentile
        bvs_shuffles           :  blob@imaging_store   #  Individual shuffles BVS 
        """

    @property
    def key_source(self):
        return super().key_source & 'bvfield_params_id="A"' \
             & 'bvscore_params_id="A"' & 'bv_field_dect_method="bvs"' \
             & 's_t_params_id = "A"'
        # Constrain to some parameters for now 

    def make(self,key):

        ''' 
        This routine follows the basic layout of "SignalTracking" (see ratemaps.py in main imaging schema).
        It extracts sync and signal data and creates a vector of shuffling integers 
        that are used to 'roll' the frames sync signal around. 
        This maintains the underlying signal statistics, and destroys the correspondence
        between signal and tracking data.
        '''    
        # Try to fetch the occupancy. 
        # Ideally we should not inherit from Tracking but instead from Occupancy - this would be cleaner
        # Right now, things run into errors if Occupancy has not been populated 

        try: 
            occupancy_entry  = (Occupancy & key).fetch1()
        except dj.DataJointError:
            print('Occupancy not found. Skipping.')
            return 


        bvs_field_method = (BVScoreFieldMethod & key).fetch1('bv_field_dect_method')
        if bvs_field_method not in ['opexebo','bvs']:
            raise NotImplementedError(f'Method "{bvs_field_method}" does not have a matching score calculation routine')


        shuffle_params = (ShuffleParams & key).fetch1()
        
        st_params = {}
        st_params['speed_cutoff_low'], st_params['speed_cutoff_high'], time_offset = (SignalTrackingParams & key).fetch1(
                                                                    'speed_cutoff_low', 'speed_cutoff_high', 'time_offset')

        events   = (FilteredEvents.proj(signal_dataset='dataset_name', events='filtered_events')
                    & key).fetch1('events')
        tracking_ = (Tracking.OpenField * Tracking.proj(tracking_dataset='dataset_name')
                    & key).fetch1()

        center_y, center_plane, proj_mean_img = ((Projection.proj('mean_image') * Cell.Rois * RoisCorr.proj('center_plane')).proj(
                    ..., signal_dataset='dataset_name') & key).fetch1('center_y', 'center_plane', 'mean_image')
        num_planes, frame_rate_si, seconds_per_line, width_SI, height_SI = (Tif.SI & (Recording & key)).fetch1(
                    'num_scanning_depths', 'framerate', 'seconds_per_line', 'width_scanimage', 'height_scanimage')

        # Special case where the SI image shape is not the same with the projection image shape
        # - suggesting the image has been cropped prior to suite2p analysis
        # - thus, the "center_y" is no longer accurate -> using the middle line for "center_y"
        if proj_mean_img.shape != (width_SI, height_SI):
            center_y = int(height_SI/2)

        # Ratemap 
        ratemap_params   = (MapParams & key).fetch1()
        #field_params     = (FieldParams & key).fetch1() # Skipped for now - opexebo field detection

        # BV Field method 
        bvs_field_params =  (BVFieldParams & key).fetch1()

        occupancy = np.ma.array(occupancy_entry['occupancy'], mask=occupancy_entry['mask_occ'])
        x_edges = occupancy_entry['x_edges'].copy()
        y_edges = occupancy_entry['y_edges'].copy()

        # Experiment Type
        equipment_type   = (Recording & key).fetch1('equipment_type')

        # Retrieve Sync
        # 1. Imaging
        sync_data_frames, sample_rate_sync, key['sync_dataset_frames_imaging'], key['sync_name_frames_imaging'] = \
                                                (Sync \
                                                & {'sync_name':'frames'} & key).fetch1(
                                                'sync_data', 'sample_rate', 'dataset_name', 'sync_name')

        # 2. Tracking
        # Is dataset deep lab cut? if yes, load as 
        # -> 'generic_name = "TrackingDLC"', otherwise
        # -> 'generic_name = "Tracking2LED"' 
        tracking_type = (Dataset & 'dataset_name = "{}"'.format(key['tracking_dataset'])).fetch1('datasettype')
        if tracking_type == 'DLC_tracking':
            tracking_generic = 'Tracking_DLC'
        elif 'Tracking2D_2LED' in tracking_type:
            tracking_generic = 'Tracking2LED'
        else:
            raise NotImplementedError(f'Tracking dataset type {tracking_type} not implemented')

        sync_data_track = (Session.SystemConfig * SystemConfig.Sync * Sync 
                           & f'sync_purpose = "{tracking_generic}"' & key).fetch1('sync_data')

        # Sanity checks
        # 1. Compare length of tracking data and tracking sync data
        # 2. Compare length of spike data and frame sync data 
        # 3. Compare last timestamp frame sync data and tracking sync data

        if len(tracking_['x_pos']) != len(sync_data_track):
            raise IndexError('Mismatch between length of sync data and tracking data')
        if len(events) != len(sync_data_frames):
            raise IndexError('Mismatch between length of sync data and spiking data')
        if np.abs(sync_data_track[-1] - sync_data_frames[-1]) > np.mean(np.diff(sync_data_frames)):
            raise IndexError('There is more than one frame difference between the end of sync streams')

        # -> Compensate for the mismatch between timestamp at the beginning of each frame and the time it takes the laser to reach the cell body
        seconds_per_plane =  1 / (frame_rate_si * num_planes) # from Tif.SI()
        # Why is this correct? Because frame_rate_si returns the "volume" rate. 

        if '2Pmini' in equipment_type:
            # Careful! "samples" are floating point (real valued) timestamps for pre-synced setups since 
            # sample rate = 1. for those sync data
            seconds_to_cell  = center_plane * seconds_per_plane + center_y * seconds_per_line
            samples_to_cell  = seconds_to_cell * sample_rate_sync 
            samples_offset   = samples_to_cell + (time_offset * sample_rate_sync) # from 'MapParams'
        else:
            raise NotImplementedError('Cell time finding not implemented for experiment type "{}""'.format(equipment_type))


        ######### CREATE SHUFFLING VECTOR ############################################################################

        margin_seconds = shuffle_params['margin_seconds'] 
        break_seconds  = shuffle_params['break_seconds']
        samples_break = np.ceil(break_seconds / (np.diff(sync_data_frames).mean()/sample_rate_sync)).astype(int)
        sample_from_start = sync_data_frames[0] + (sample_rate_sync * margin_seconds)
        sample_from_end   = sync_data_frames[-1] - (sample_rate_sync * margin_seconds)
        range_start = find_nearest(sync_data_frames, sample_from_start)
        range_end   = find_nearest(sync_data_frames, sample_from_end)

        population_rolling_idxs = range(range_start, range_end, samples_break)
        number_of_shuffles = min(shuffle_params['number_shuffles'], len(population_rolling_idxs))
        shuffling_offsets = random.sample(population_rolling_idxs, number_of_shuffles)


        ######### INSERT INTO MASTER TABLE ##########################################################################    
        key_ = {
            'number_shuffles'  : len(shuffling_offsets), 
            'shuffling_offsets': shuffling_offsets
            }
        self.insert1({**key, **key_})

        # Clean up key
        _ = key.pop('sync_dataset_frames_imaging', None)
        _ = key.pop('sync_name_frames_imaging', None)


        ######### PREPARE SIGNAL AND SHUFFLES ########################################################################

        # Only do this for events right now
        signal_data   =  events
        signal_idxs   =  np.argwhere(events > 0).squeeze()

        # Shuffled array
        shuffled_bvs = []

        ######### SHUFFLE ###########################################################################################

        for shift in shuffling_offsets: # tqdm(shuffling_offsets):
            rolled_sync_data_frames = np.roll(sync_data_frames, shift)

            # Look up signal tracking
            signal_dict = calc_signal_dict(rolled_sync_data_frames, sync_data_track,
                                           tracking_, signal_data, signal_idxs, samples_offset, st_params)

            # Get ratemap
            try:
                ratemap_dict = calc_ratemap(occupancy, x_edges, y_edges, signal_dict, ratemap_params)
            except IndexError:  # This happens if only one signal value was found overall
                continue
            
            ratemap_nans =  np.ma.filled(ratemap_dict['ratemap'], fill_value=np.nan).astype(np.float64)
            try:
                if bvs_field_method == 'opexebo':
                    raise NotImplementedError('Field detection method opexebo is not implemented in shuffling')
                    # _, fields_map = opexebo.analysis.place_field(
                    #                     ratemap_nans, min_bins=field_params['min_bins'],
                    #                     min_mean=ratemap_dict['ratemap'].max()*field_params['fraction_min_mean'],
                    #                     min_peak=ratemap_dict['ratemap'].max()*field_params['fraction_min_peak'],
                    #                     init_thresh=field_params['init_thresh'], search_method=field_params['search_method'])

                    # fieldmap = fields_map.copy()
                    # fieldmap[fieldmap>0] = 1
                elif bvs_field_method == 'bvs':
                    fieldmap, _ = detect_fields(ratemap_nans, minBin=bvs_field_params['min_bin'], \
                                                            std_include=bvs_field_params['std_include'], \
                                                            std_detect=bvs_field_params['std_detect'],\
                                                            show_plots=False, debug=False)
            except: 
                # Whatever, just catch for now ... 
                continue 
                

            if (fieldmap==0).all():
                # Empty fieldmap! 
                continue 

            bvs_params = (BVScoreParams & key).fetch1()
            bvs, _, _, _ = calc_bv_score(fieldmap, r=bvs_params['r_factor'], \
                                        barwidth_max=bvs_params['barwidth_max'], show_plots=False, debug=False)
            shuffled_bvs.append(bvs)


        ######### FILL PART TABLES ################################################################################## 
    
        BVS_Score_dict = {
            'bvs_99'                :  np.nanpercentile(shuffled_bvs, 99),
            'bvs_95'                :  np.nanpercentile(shuffled_bvs, 95),
            'bvs_shuffles'          :  np.array(shuffled_bvs).astype(float)
        }
        self.BVS.insert1({**key, **BVS_Score_dict}, ignore_extra_fields=True)
        