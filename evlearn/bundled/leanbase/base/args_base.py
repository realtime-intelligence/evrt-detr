import difflib
import os

LABEL_FNAME = 'label'
ENCODING    = 'utf-8'

def get_config_difference(config_old, config_new):
    diff_gen = difflib.unified_diff(
        config_old.to_json(sort_keys = True, indent = 4).split('\n'),
        config_new.to_json(sort_keys = True, indent = 4).split('\n'),
        fromfile = 'Old Config',
        tofile   = 'New Config',
    )

    return "\n".join(diff_gen)

class ArgsBase:
    config_cls = lambda x : None

    __slots__ = [
        'config',
        'label',
        'savedir',
    ]

    def __init__(self, config, savedir, label):
        self.config  = config
        self.label   = label
        self.savedir = savedir

    def __getattr__(self, attr):
        return getattr(self.config, attr)

    def _save_extras(self):
        pass

    def _load_extras(self):
        pass

    def save(self):
        self.config.save(self.savedir)

        if self.label is not None:
            label_path = os.path.join(self.savedir, LABEL_FNAME)
            with open(label_path, 'wt', encoding = ENCODING) as f:
                f.write(self.label)

        self._save_extras()

    def check_no_collision(self):
        try:
            old_config = self.config_cls.load(self.savedir)
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
                )
            )

    @classmethod
    def _from_args_dict(
        cls, config_dict, outdir, label = None, **kwargs
    ):
        config  = cls.config_cls(**config_dict)
        savedir = config.get_savedir(outdir, label)

        result = cls(config, savedir, label, **kwargs)
        result.check_no_collision()

        result.save()

        return result

    @classmethod
    def load(cls, savedir):
        config = cls.config_cls.load(savedir)
        label  = None

        label_path = os.path.join(savedir, LABEL_FNAME)

        if os.path.exists(label_path):
            # pylint: disable=unspecified-encoding
            with open(label_path, 'rt', encoding = ENCODING) as f:
                label = f.read()

        result = cls(config, savedir, label)
        result._load_extras()

        return result

