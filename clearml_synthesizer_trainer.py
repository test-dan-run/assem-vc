from clearml import Task, Dataset, Model

task = Task.init(project_name='voice_conversion/vc_decoder', task_name='vc_decoder_train', output_uri='s3://experiment-logging/storage', task_type='training')
task.set_base_docker('dleongsh/assem-vc:v1.8.1')
task.execute_remotely(queue_name='compute', clone=False, exit_process=True)

GLOBAL_CONFIG_PATH = 'config/global/config.yaml'
VC_CONFIG_PATH = 'config/vc/config.yaml'
NAME = 'vc_decoder_train'
COTA_MODEL_ID = '3437c077c4434b3fbed68157fec05478'

import os
import warnings
warnings.simplefilter("ignore", UserWarning)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from utils.utils import get_commit_hash
from synthesizer import Synthesizer
from utils.loggers import SynthesizerLogger

def main():

    hp_global = OmegaConf.load(GLOBAL_CONFIG_PATH)
    hp_vc = OmegaConf.load(VC_CONFIG_PATH)

    # download data
    data = Dataset.get(dataset_project=hp_global.data.dataset_project, dataset_name=hp_global.data.dataset_name)
    dataset_path = data.get_local_copy()
    hp_global.data.train_dir = dataset_path
    hp_global.data.val_dir = dataset_path
    hp_global.data.f0s_list_path = os.path.join(dataset_path, 'f0s.txt')

    cota_model = Model(model_id = COTA_MODEL_ID)
    hp_vc.train.cotatron_path = cota_model.get_local_copy()

    hp = OmegaConf.merge(hp_global, hp_vc)
    model = Synthesizer(hp)
    model.load_cotatron(hp.train.cotatron_path)

    save_path = os.path.join(hp.log.chkpt_dir, NAME)
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(hp.log.chkpt_dir, NAME, get_commit_hash()+'_best'),
        monitor='val_loss',
        verbose=True,
        save_top_k=1,
    )

    tb_logger = SynthesizerLogger(
        save_dir=hp.log.log_dir,
        name=NAME,
    )

    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        default_root_dir=save_path,
        gpus=-1,
        accelerator=None,
        num_sanity_val_steps=1,
        resume_from_checkpoint=None,
        gradient_clip_val=0.0,
        progress_bar_refresh_rate=10,
        max_epochs=10000,
    )
    trainer.fit(model)

if __name__ == '__main__':
    main()
