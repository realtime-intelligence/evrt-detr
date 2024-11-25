import difflib
import os

from .config import Config
from ..consts import ROOT_OUTDIR

LABEL_FNAME = 'label'
CONFIG_NAME = 'config.json'

def get_config_difference(config_old, config_new):
    diff_gen = difflib.unified_diff(
        config_old.to_json(sort_keys = True, indent = 4).split('\n'),
        config_new.to_json(sort_keys = True, indent = 4).split('\n'),
        fromfile = 'Old Config',
        tofile   = 'New Config',
    )

    return "\n".join(diff_gen)

class Args:
    __slots__ = [
        'config',
        'label',
        'savedir',
        'checkpoint',
    ]

    def __init__(
        self, config, savedir, label,
        checkpoint = 100,
    ):
        # pylint: disable=too-many-arguments
        self.config     = config
        self.label      = label
        self.savedir    = savedir
        self.checkpoint = checkpoint

    def __getattr__(self, attr):
        return getattr(self.config, attr)

    def save(self):
        self.config.save(self.savedir, CONFIG_NAME)

        if self.label is not None:
            # pylint: disable=unspecified-encoding
            with open(os.path.join(self.savedir, LABEL_FNAME), 'wt') as f:
                f.write(self.label)

    def check_no_collision(self):
        try:
            old_config = Config.load(self.savedir, CONFIG_NAME)
        except IOError:
            return

        old = old_config.to_json(sort_keys = True)
        new = self.config.to_json(sort_keys = True)

        if old != new:
            diff = get_config_difference(old_config, self.config)

            raise RuntimeError(
                (
                    f"Config collision detected in '{self.savedir}'"
                    f" . Difference:\n{diff}"
                    "\n\n[NOTE] If you really want to overwrite the model,"
                    f" remove the 'config.json' file in '{self.savedir}'\n"
                )
            )

    @staticmethod
    def from_args_dict(
        outdir,
        label      = None,
        checkpoint = 100,
        **args_dict
    ):
        config  = Config(**args_dict)

        outdir  = os.path.join(ROOT_OUTDIR, outdir)
        savedir = config.get_savedir(outdir, label)

        result = Args(config, savedir, label, checkpoint)
        result.check_no_collision()

        result.save()

        return result

    @staticmethod
    def load(savedir):
        config = Config.load(savedir, CONFIG_NAME)
        label  = None

        label_path = os.path.join(savedir, LABEL_FNAME)

        if os.path.exists(label_path):
            # pylint: disable=unspecified-encoding
            with open(label_path, 'rt') as f:
                label = f.read()

        return Args(config, savedir, label)

