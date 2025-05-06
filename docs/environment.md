# Environment Configuration

## Create a new python environment
In Ubuntu 24.04 this is necessary to install python packages using pip:
```bash
python3 -m venv myenvTesis
source myenvTesis/bin/activate
pip install package_name

deactivate
```

## numpy version < 2.0.0
ATTENTION! numpy version < 2.0.0 needed, if not, it will fail with the following error:
> OverflowError: Python integer -20 out of bounds for uint8

```bash
pip install numpy==1.26.4
```

## Analyzing loss curve after model training
In terminal, change path to where the log file exists, then:

```bash
pip uninstall numpy
pip install numpy==1.26.4
source envTensorboard/bin/activate
tensorboard --logdir=.

deactivate
```

If getting an error when training the model about "tensorflow not installed" derived from tensorboard, just "pip uninstall tensorflow" and then reinstall tensorboard-logger.
