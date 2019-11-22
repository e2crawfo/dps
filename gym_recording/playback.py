import os
import glob
import logging
import dill
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


__all__ = ['scan_recorded_traces', 'TraceRecordingReader']


class TraceRecordingReader:
    def __init__(self, directory):
        self.directory = directory
        self.binfiles = {}

    def close(self):
        for k in self.binfiles.keys():
            if self.binfiles[k] is not None:
                self.binfiles[k].close()
                self.binfiles[k] = None

    def get_recorded_batches(self):
        ret = []
        manifest_ptn = os.path.join(self.directory, 'openaigym.trace.*.manifest.pkl')
        trace_manifest_fns = glob.glob(manifest_ptn)
        logger.debug('Trace manifests %s %s', manifest_ptn, trace_manifest_fns)

        for trace_manifest_fn in trace_manifest_fns:
            with open(trace_manifest_fn, 'rb') as f:
                trace_manifest = dill.load(f)

            ret.extend(trace_manifest['batches'])
        return ret

    def get_recorded_episodes(self, batch):
        filename = os.path.join(self.directory, batch['fn'])
        with open(filename, 'rb') as f:
            batch_d = dill.load(f)
        return batch_d['episodes']


def scan_recorded_traces(directory, episode_cb=None, max_episodes=None, episode_range=None):
    """
    Go through all the traces recorded to directory, and call episode_cb for every episode.
    Set max_episodes to end after a certain number (or you can just throw an exception from episode_cb
    if you want to end the iteration early)
    """
    rdr = TraceRecordingReader(directory)

    recorded_batches = rdr.get_recorded_batches()

    all_episodes = [ep for batch in recorded_batches for ep in rdr.get_recorded_episodes(batch)]

    if episode_range:
        all_episodes = all_episodes[slice(*episode_range)]

    print("Found {} episodes (after applying episode range).".format(len(all_episodes)))
    ep_lengths = [len(ep['rewards']) for ep in all_episodes]
    print("Episode length stats:")
    print(pd.DataFrame(ep_lengths).describe())
    print("Total number of observations: {}".format(sum(ep_lengths)))

    if max_episodes is None or max_episodes >= len(all_episodes):
        episodes = all_episodes
    else:
        episode_indices = np.random.choice(len(all_episodes), size=max_episodes, replace=False)
        episodes = [all_episodes[i] for i in episode_indices]

    for ep in episodes:
        done = episode_cb(ep['observations'], ep['actions'], ep['rewards'])
        if done:
            break

    rdr.close()
