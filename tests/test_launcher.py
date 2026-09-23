import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'nt', 'Windows BAT integration')
class LauncherTests(unittest.TestCase):
    def run_launcher(self, argument):
        env = os.environ.copy()
        env['COST_LAUNCH_FILE'] = str(ROOT / '启动成本测算.bat')
        return subprocess.run(
            'cmd.exe /d /c call "%COST_LAUNCH_FILE%" ' + argument,
            env=env, cwd=str(ROOT.parent), capture_output=True, timeout=55,
        )

    def test_check_mode_validates_single_entry_from_another_directory(self):
        result = self.run_launcher('--check')
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', errors='replace'))
        output = result.stdout.decode('utf-8', errors='replace')
        self.assertIn('app\\成本测算.py', output)
        self.assertIn('火电机组成本测算研究原型', output)
        self.assertIn('Environment check passed', output)

    def test_public_launcher_has_no_personal_conda_environment_name(self):
        launcher = (ROOT / '启动成本测算.bat').read_text(encoding='utf-8')
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertNotIn('powerdas', launcher.lower())
        self.assertNotIn('powerdas', readme.lower())


if __name__ == '__main__':
    unittest.main()
