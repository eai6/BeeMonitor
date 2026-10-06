"""pybioclip must not switch torch to weights-only loading for the worker."""

import os
import subprocess
import sys
import unittest


class WeightsOnlyEnvTests(unittest.TestCase):
    def test_importing_the_wrapper_keeps_full_checkpoint_loading(self):
        # A fresh process: the env var is set at import time, once.
        env = {k: v for k, v in os.environ.items() if k != "TORCH_FORCE_WEIGHTS_ONLY_LOAD"}
        out = subprocess.run(
            [sys.executable, "-c",
             "import os, beemonitor.identification.bioclip; "
             "print(os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD'))"],
            env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "0")


if __name__ == "__main__":
    unittest.main()
