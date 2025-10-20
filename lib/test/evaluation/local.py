from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/workspace/dyf/code/0924_0603_138/data/got10k_lmdb'
    settings.got10k_path = '/root/workspace/dyf/code/0924_0603_138/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/root/workspace/dyf/code/0924_0603_138/data/itb'
    settings.lasot_extension_subset_path_path = '/root/workspace/dyf/code/0924_0603_138/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/root/workspace/dyf/code/0924_0603_138/data/lasot_lmdb'
    settings.lasot_path = '/root/workspace/dyf/code/0924_0603_138/data/lasot'
    settings.network_path = '/root/workspace/dyf/code/0924_0603_138/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/workspace/dyf/code/0924_0603_138/data/nfs'
    settings.otb_path = '/root/workspace/dyf/code/0924_0603_138/data/otb'
    settings.prj_dir = '/root/workspace/dyf/code/0924_0603_138'
    settings.result_plot_path = '/root/workspace/dyf/code/0924_0603_138/output/test/result_plots'
    settings.results_path = '/root/workspace/dyf/code/0924_0603_138/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/workspace/dyf/code/0924_0603_138/output'
    # settings.save_dir = '/root/workspace/output'
    settings.segmentation_path = '/root/workspace/dyf/code/0924_0603_138/output/test/segmentation_results'
    settings.tc128_path = '/root/workspace/dyf/code/0924_0603_138/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/root/workspace/dyf/code/0924_0603_138/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/workspace/dyf/code/0924_0603_138/data/trackingnet'
    settings.uav_path = '/root/workspace/dyf/code/0924_0603_138/data/uav'
    settings.vot18_path = '/root/workspace/dyf/code/0924_0603_138/data/vot2018'
    settings.vot22_path = '/root/workspace/dyf/code/0924_0603_138/data/vot2022'
    settings.vot_path = '/root/workspace/dyf/code/0924_0603_138/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

