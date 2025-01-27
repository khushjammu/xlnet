## IMDB 
### Experimentation Phase 1

The command I used for phase 1. 

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=/home/jammu55048/aclImdb \
  --output_dir=${GS_ROOT}/proc_data/imdb \
  --model_dir=${GS_ROOT}/exp/imdb \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint==${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500
```


### Experimentation Phase 2

The command I used for phase 2. I assumed in my code that `model_dir` is the same as the output dir for my profiling. Let's see if that's an accurate assumption. 

#### Trial 1

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=/home/jammu55048/aclImdb \
  --output_dir=${GS_ROOT}/proc_data/imdb \
  --model_dir=${GS_ROOT}/exp/imdb \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500
```

#### Trial 2

Changed paths slightly to make backing stuff up easier.

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=/home/jammu55048/aclImdb \
  --output_dir=${GS_ROOT}/experimentation_phase_2/trial_2/proc_data/imdb \
  --model_dir=${GS_ROOT}/experimentation_phase_2/trial_2/exp/imdb \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500 2>&1 | tee command_output.txt
```

#### Trial 3,4,5 (just replace number in file directories)

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=/home/jammu55048/aclImdb \
  --output_dir=${GS_ROOT}/experimentation_phase_2/trial_3/proc_data/imdb \
  --model_dir=${GS_ROOT}/experimentation_phase_2/trial_3/exp/imdb \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500 2>&1 | tee command_output.txt
```

*Trial five worked! The amount of samples it takes is slow but it fucking works.*

#### Trial 6,7 

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=/home/jammu55048/aclImdb \
  --output_dir=${GS_ROOT}/experimentation_phase_2/trial_6/proc_data/imdb \
  --model_dir=${GS_ROOT}/experimentation_phase_2/trial_6/exp/imdb \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500 2>&1 | tee command_output.txt
```

Trying to increase the number of profiles now. General experimentation. 

#### Trial 8,9

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=/home/jammu55048/aclImdb \
  --output_dir=${GS_ROOT}/experimentation_phase_2/trial_8/proc_data/imdb \
  --model_dir=${GS_ROOT}/experimentation_phase_2/trial_8/exp/imdb \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=100 \
  --iterations=100 2>&1 | tee command_output.txt
```

It's taking too long. I'm cutting down the number of iterations.

~~*FAILED.* Couldn't get it working. Whatever — will use my current way of doing it.~~

#### Trial 10

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=/home/jammu55048/aclImdb \
  --output_dir=${GS_ROOT}/experimentation_phase_2/trial_10/proc_data/imdb \
  --model_dir=${GS_ROOT}/experimentation_phase_2/trial_10/exp/imdb \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=100 \
  --iterations=100 2>&1 | tee command_output.txt
```

Turns out, it might not have been working b/c I forgot to comment out the code in another document. 

*Also didn't work, possibly because of how I've implemented it. I suspect it's the timer. In any case, I don't really care. Let's just get it working. *

#### Trial 11

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir=/home/jammu55048/aclImdb \
  --output_dir=${GS_ROOT}/experimentation_phase_2/trial_11/proc_data/imdb \
  --model_dir=${GS_ROOT}/experimentation_phase_2/trial_11/exp/imdb \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500 2>&1 | tee command_output.txt
```

The final shot. 

I FUCKED IT UP by not pushing the commit which added back in the profiler, so the whole run was wasted. I'm not going to re-run it now: I just need one profile. Better to get SQuAD done first. 

## SQuAD
### Experimentation Phase 3

Now that IMDB is working, time for SQuAD. I preprocessed the data by modifying `prepro_squad.sh` as follows:

```bash
#!/bin/bash

#### local path
SQUAD_DIR=/home/jammu55048/data/squad
INIT_CKPT_DIR=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16

#### google storage path
GS_ROOT=gs://khush_ee
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/squad

python run_squad.py \
  --use_tpu=False \
  --do_prepro=True \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --train_file=${SQUAD_DIR}/train-v2.0.json \
  --output_dir=${GS_PROC_DATA_DIR} \
  --uncased=False \
  --max_seq_length=512 \
  --num_proc=4
  $@
```

When copying and pasting the above, I saw the option for multi-processing. I feel like an idiot. Next time, read the whole script before running it. 

Anyway, now to try and get the finetuning for SQuAD working. 

#### Trial 1

The bash script `tpu_squad_large.sh` just runs the python script `run_squad.py` with some parameters. Way easier just to run it directly — much more control.


```bash
python run_squad.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --model_config_path=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --output_dir=${GS_ROOT}/proc_data/squad \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --model_dir=${GS_ROOT}/experimentation_phase_3/experiment/squad \
  --train_file=/home/jammu55048/xlnet/data/squad/train-v2.0.json \
  --predict_file=/home/jammu55048/xlnet/data/squad/dev-v2.0.json \
  --uncased=False \
  --max_seq_length=512 \
  --do_train=True \
  --train_batch_size=48 \
  --do_predict=True \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=1000 \
  --train_steps=8000 \
  --warmup_steps=1000 \
  $@ 2>&1 | tee command_output.txt
```

I need to determine whether or not the script writes anything to the `output_dir`, which might seem stupid but that directory is where the preprocessed SQuAD data lives, so it'll be strange if it's written to. For future reference:

```shell
jammu55048@ctpu-cli:~/xlnet$ gsutil ls ${GS_ROOT}/proc_data/squad
gs://khush_ee/proc_data/squad/
gs://khush_ee/proc_data/squad/spiece.model.0.slen-512.qlen-64.train.tf_record
```

This worked fine. It started training. Next step: integrating profiler. To do so I'll return to previous stage. Note: I forgot to save it to a specific trial folder so the file structure is a bit messy. To be fixed. 

#### Trial 2

Going to try it with profiler now. Here goes. 
```bash
python run_squad.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --model_config_path=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --output_dir=${GS_ROOT}/proc_data/squad \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --model_dir=${GS_ROOT}/experimentation_phase_3/trial_2/experiment/squad \
  --train_file=/home/jammu55048/xlnet/data/squad/train-v2.0.json \
  --predict_file=/home/jammu55048/xlnet/data/squad/dev-v2.0.json \
  --uncased=False \
  --max_seq_length=512 \
  --do_train=True \
  --train_batch_size=48 \
  --do_predict=True \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=1000 \
  --train_steps=8000 \
  --warmup_steps=1000 \
  $@ 2>&1 | tee command_output.txt
```

Worked awesomely!

## GLUE
### Experimentation Phase 4

I want one more test that works so I can say I've tested a wide variety. 


#### Trial 1

Command I used for training. 

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=False \
  --task_name=sts-b \
  --data_dir=/home/jammu55048/xlnet/glue_data/STS-B \
  --output_dir=gs://khush_ee/experimentation_phase_4/trial_1/proc_data/sts-b \
  --model_dir=gs://khush_ee/experimentation_phase_4/trial_1/exp/sts-b \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=5e-5 \
  --train_steps=1200 \
  --warmup_steps=120 \
  --save_steps=600 \
  --is_regression=True 2>&1 | tee command_output.txt
```

Like a dumbass, I forgot to make the eval flag so I went ahead and used below for eval. 

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=False \
  --do_eval=True \
  --task_name=sts-b \
  --data_dir=/home/jammu55048/xlnet/glue_data/STS-B \
  --output_dir=gs://khush_ee/experimentation_phase_4/trial_1/proc_data/sts-b \
  --model_dir=gs://khush_ee/experimentation_phase_4/trial_1/exp/sts-b \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --max_seq_length=128 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --eval_all_ckpt=True \
  --is_regression=True 2>&1 | tee command_output.txt
```

#### Trial 2

Profiler only ran once and that too when it wasn't using the TPU smh. I'll just run the profiler manually. 

```bash
python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=False \
  --task_name=sts-b \
  --data_dir=/home/jammu55048/xlnet/glue_data/STS-B \
  --output_dir=gs://khush_ee/experimentation_phase_4/trial_2/proc_data/sts-b \
  --model_dir=gs://khush_ee/experimentation_phase_4/trial_2/exp/sts-b \
  --uncased=False \
  --spiece_model_file=/home/jammu55048/xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=5e-5 \
  --train_steps=1200 \
  --warmup_steps=120 \
  --save_steps=600 \
  --is_regression=True 2>&1 | tee command_output.txt
```

Command for manual profiling: 
```bash
capture_tpu_profile --tpu=$TPU_NAME --logdir=gs://khush_ee/experimentation_phase_4/trial_2/exp/sts-b
```

Worked good enough. Time to work on BERT now. 



