# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import datasets
from dataclasses import dataclass, field

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

logger = logging.getLogger(name=__name__)

@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the SFT training script.

    Args:
        cache_dir (`str`):
            Directory to cache datasets (optional).
        validation_split (`float`):
            Proportion of the dataset to use for validation (0.0 to 1.0).
        max_train_samples (`int`):
            Maximum number of training samples (optional).
        max_eval_samples (`int`):
            Maximum number of evaluation samples (optional).
        resume (`str`):
            Path to a checkpoint to resume training from (optional).
    """
    cache_dir: str = field(
        default="/pri_exthome/Mamba/HuggingFace_Cache",
        metadata={"help": "Directory to cache datasets (optional)"},
    )
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Proportion of the dataset to use for validation (0.0 to 1.0)"},
    )
    max_train_samples: int = field(
        default=None,
        metadata={"help": "Maximum number of training samples (optional)"},
    )
    max_eval_samples: int = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples (optional)"},
    )
    resume: bool = field(
        default=False,
        metadata={"help": "Path to a checkpoint to resume training from (optional)"},
    )

SYSTEM_PROMPT = \
    (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process are enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think>"
    )


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if script_args.resume:
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir=script_args.cache_dir)
    
    # OpenR1-Math-220k
    if "OpenR1-Math-220k" in script_args.dataset_name:
        logger.info("*** Loading OpenR1-Math-220k ***")   
        
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
                "completion": [{"role": "assistant", "content": example["generations"][0].strip()}]
            }
        dataset['default'] = dataset['default'].map(make_conversation) 
          
        if (script_args.validation_split is not None) and training_args.do_eval:
            logger.info("*** Spliting Train and Eval ***") 
            split_dataset = dataset['default'].train_test_split(test_size=script_args.validation_split, shuffle=False)
            train_dataset = split_dataset[script_args.dataset_train_split]
            eval_dataset = split_dataset[script_args.dataset_test_split]
        else:
            train_dataset = dataset['default']
            eval_dataset = None

    if training_args.do_train:
        if train_dataset is None:
            raise ValueError("--do_train requires a train dataset")
        if script_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), script_args.max_train_samples)
            logger.info(f"Max train samples set to: {max_train_samples}")
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if eval_dataset is None:
            raise ValueError("--do_eval requires a valid dataset")
        if script_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), script_args.max_eval_samples)
            logger.info(f"Max eval samples set to: {max_eval_samples}")
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path, **model_kwargs)
    model.config.max_position_embeddings = 131072

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
