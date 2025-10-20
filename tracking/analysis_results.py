import _init_paths
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'

trackers.extend(trackerlist(name='gfatrack', parameter_name='vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='gfatrack256'))



dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

