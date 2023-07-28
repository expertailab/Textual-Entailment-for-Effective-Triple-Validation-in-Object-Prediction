===============================
Textual Entailment for Effective Triple Validation in Object Prediction
===============================

|PyPI pyversions|

Code developed with 💛 at `Expert.ai Research Lab <https://expert.ai>`__ for the paper Textual Entailment for Effective Triple Validation in Object Prediction.

Installation
------------

The whole project is handled with ``make``, go to a terminal an issue:

.. code:: bash

   git clone https://github.com/expertailab/Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction.git
   cd Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction
   make setup
   conda activate lm_kbc
   make install-as-pkg

Reproducibility
---------------

Important note: For each experiment we will generate a predictions file that will be use to get the final evaluation results. To get the results we have to run:

.. code:: bash

   python lm_kbc/evaluating/evaluate.py -g data/raw/lm-kbc/dataset/data/dev.jsonl -p $PREDICTIONS_FILE

Where $PREDICTIONS_FILE is the path to the predictions.

Pretrained models experiments
~~~~~~~~~~~~~~~~~~~~~
**Language model baseline**

We can run the language model baseline using a BERT large with:

.. code:: bash

   python lm_kbc/modeling/zero_shot_entailment.py --is_baseline --candidates_generation from_lm --lm_candidates bert-large-cased --filter_before --calculate_lm_threshold --input_path_dev_2 data/raw/lm-kbc/dataset/data/train.jsonl --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl  --output_path lm_baseline.jsonl

**Question answering baseline**

To run the question answering baseline, we need the passages to find the answers for the questions generated with the development set and training set, we can obtain them using the `get_contexts.py <scripts/get_contexts.py>`_ script:

.. code:: bash

   python scripts/get_contexts.py --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl --contexts_path ./contexts.json
   python scripts/get_contexts.py --input_path ./data/raw/lm-kbc/dataset/data/train.jsonl --contexts_path ./contexts_train.json

However, we can download the already created contexts with:

.. code:: bash

   wget https://zenodo.org/record/7624717/files/contexts.json
   wget https://zenodo.org/record/7624717/files/contexts_train.json


Now we can run the question answering baseline using a DeBERTa large model fine-tuned on SQuAD v2 with:

.. code:: bash

   python lm_kbc/modeling/zero_shot_qa.py --model deepset/deberta-v3-large-squad2 --contexts_path contexts.json --calculate_qa_threshold --contexts_train_path contexts_train.json  --input_path_dev_2 ./data/raw/lm-kbc/dataset/data/train.jsonl --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl --output_path qa_baseline.jsonl

**Relation extraction baseline**

We can run the relation extraction baseline using a REBEL large with:

.. code:: bash

   python lm_kbc/modeling/zero_shot_rebel.py --model Babelscape/rebel-large --input_path data/raw/lm-kbc/dataset/data/dev.jsonl --contexts_path contexts.json --output_path rebel_baseline.jsonl

**SATORI**

We can use SATORI (Seek and enTail for Object pRedIction) with different pretrained entailment models (we have tried with DeBERTa xsmall, BERT large, and a DeBERTa xlarge fine-tuned on NLI/MNLI datasets), but also with different object sources:

* Like a BERT large as object source:

   .. code:: bash

      python lm_kbc/modeling/zero_shot_entailment.py --candidates_generation from_lm --lm_candidates bert-large-cased --calculate_lm_threshold --input_path_dev_2 ./data/raw/lm-kbc/dataset/data/train.jsonl --contexts_train_path contexts_train.json --filter_before --filter_fixed_candidates --model cross-encoder/nli-deberta-v3-xsmall --contexts_path contexts.json --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl --output_path satori-deberta-xsmall-from_lm-calculate-thresholds.jsonl

* A combination of objects from contexts (NER) and fixed candidates (KG):

   .. code:: bash

      python lm_kbc/modeling/zero_shot_entailment.py --candidates_generation from_contexts --use_candidates_fixed --calculate_entailment_threshold --input_path_dev_2 ./data/raw/lm-kbc/dataset/data/train.jsonl --contexts_train_path contexts_train.json --filter_fixed_candidates --model cross-encoder/nli-deberta-v3-xsmall --contexts_path contexts.json --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl --output_path satori-deberta-xsmall-from_contexts_and_fixed-calculate_entailment_threshold.jsonl

* Or we can use as object source a merge of all the sources (from LM, from contexts (NER), and using fixed candidates (KG)):

   .. code:: bash

      python lm_kbc/modeling/zero_shot_entailment.py --candidates_generation merge --lm_candidates bert-large-cased --filter_before --calculate_lm_threshold --input_path_dev_2 ./data/raw/lm-kbc/dataset/data/train.jsonl --contexts_train_path contexts_train.json --use_candidates_fixed --filter_fixed_candidates --model cross-encoder/nli-deberta-v3-xsmall --contexts_path contexts.json --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl --output_path satori-deberta-xsmall-merge-calculate_thresholds.jsonl

We can change the *--model* parameter to use other entailment model, such as BERT large fine-tuned on MNLI (boychaboy/MNLI_bert-large-cased) or DeBERTa xlarge fine-tuned on MNLI (microsoft/deberta-v2-xlarge-mnli)

Additional training experiments
~~~~~~~~~~~~~~~~~~~~

For the additional training experiments, we split the training set using 80% for "train2" set and 20% for "dev2" set. This can be done with (This is not required as we already provide these splits):

.. code:: bash

   python scripts/split_train_set.py

This will create the splits train2.jsonl and dev2.jsonl and will be at "data/processed/train/". Now we get samples of these splits using the few-shot percentages: 5, 10, 20. We can do this with (again, this is not required as we provide the samples):

.. code:: bash

   python scripts/fewshot-samples.py

This will create the files train2-$PERCENTAGE-$SAMPLE.jsonl and dev2-$PERCENTAGE-$SAMPLE.jsonl at "data/processed/train/". There will be 10 samples per each percentage.

**Language model baseline**

Here we detail how to further pre-train BERT large using Masked Language Model (MLM) task in a few-shot regime. The following are the steps to train de LM with 5% of the dataset using one of the 10 samples, this can be adapted to train the LM with other percentage or sample.

.. code:: bash

   cd ..
   git clone https://github.com/Teddy-Li/LMKBC-Track1.git
   cd LMKBC-Track1/
   conda create -n lmkbc_track1 python=3.10
   conda activate lmkbc_track1
   pip install -r requirements.txt
   mkdir data
   ln -s $(dirname $(pwd))/lm-kbc/data/processed/train/train2-5-0.jsonl data/train.jsonl
   mkdir thresholds
   mkdir outputs
   cp ../Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction/scripts/trial_1_2.py .
   python trial_1_2.py -m bert-large-cased --version baseline --job_name search_thres --subset train --comments _withsoftmax_multilm --use_softmax 1 --gpu 0 --prompt_esb_mode cmb
   ln -s $(dirname $(pwd))/Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction/data/processed/train/dev2-5-0.jsonl data/dev.jsonl
   cp ../Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction/scripts/train_mlm.py .
   python train_mlm.py --job_name collect_data --model_name bert-large-cased --top_k 100 --collect_data_gpu_id 0 --prompt_style trial --use_softmax --thresholds_fn_feat baseline_withsoftmax_multilm
   python train_mlm.py --job_name train --model_name bert-large-cased --data_mode submission --lr 5e-6 --num_epochs 10 --extend_len 0 --comment _lr5e-6_10_0 --data_suffix _baseline_withsoftmax_multilm --ckpt_dir ./models/lmkbc_checkpoints/mlm_checkpoints-005-0%s

Further pre-trained model will be stored at "./models/lmkbc_checkpoints/mlm_checkpoints-005-0_baseline_withsoftmax_multilm_lr5e-6_10_0_submission/best_ckpt/"

Once we have further pre-trained the LM, we can run the LM baseline using a 5% of the training set with:

.. code:: bash

   conda activate lm_kbc
   cd ../Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction
   python lm_kbc/modeling/zero_shot_entailment.py --is_baseline --candidates_generation from_lm --lm_candidates $(dirname $(pwd))/LMKBC-Track1/models/lmkbc_checkpoints/mlm_checkpoints-005-0_baseline_withsoftmax_multilm_lr5e-6_10_0_submission/best_ckpt/ --filter_before --calculate_lm_threshold --input_path_dev_2 ./data/processed/train/train-5-0.jsonl --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl  --output_path dev-few_shot-baseline-5-0-from_lm-calculate_lm_threshold_with_train-stopwords.jsonl

**Question answering baseline**

The steps to run the question answering baseline are the following:

#. (Optional, since we provide the additional training dataset samples in SQuAD format) To further fine-tune a question answering model, we need to create a question answering dataset from the LM KBC dataset. We have prepared the script `lmkbc2squad_fewshot.py <lm_kbc/processing/lmkbc2squad_fewshot.py>`_ for this. The script needs the LM KBC dataset few-shot and full training samples, and the contexts to find the answers to the questions in the training set. The question answering conversion script expects the contexts to be in "data/processed/train/contexts/contexts_train.json", so we copy the contexts there:

   .. code:: bash

      mkdir data/processed/train/contexts/
      cp contexts_train.json data/processed/train/contexts/

   Now we can convert the dataset to a question answering format using the contexts with:

   .. code:: bash

      python lm_kbc/processing/lmkbc2squad_fewshot.py

   The converted files will be at "data/processed/train/lm_kbc_train2_squad_$PERCENTAGE-$SAMPLE.json"

#. We further fine-tune the question answering model using the SQuAD version of our dataset, in this this example we use one of the samples of the 5% training set split:

   .. code:: bash

      conda create -n transformers python=3.10
      conda activate transformers
      conda install -c huggingface transformers==4.24.0
      conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
      pip install datasets evaluate
      wget https://github.com/huggingface/transformers/raw/v4.24.0/examples/pytorch/question-answering/utils_qa.py -P scripts
      wget https://github.com/huggingface/transformers/raw/v4.24.0/examples/pytorch/question-answering/trainer_qa.py -P scripts
      python scripts/run_qa.py --model_name_or_path deepset/deberta-v3-large-squad2 --do_train --per_device_train_batch_size 1 --gradient_accumulation_steps 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --version_2_with_negative --train_file data/processed/train/lm_kbc_train2_squad_5-0.json --output_dir ./models/lm_kbc_train2_squad_5-0

#. And we can evaluate the question answering baseline with additional training:

   .. code:: bash

      conda activate lm_kbc
      python lm_kbc/modeling/zero_shot_qa.py --model $(pwd)/models/lm_kbc_train2_squad_5-0/ --contexts_path contexts.json --calculate_qa_threshold --contexts_train_path contexts_train.json --input_path_dev_2 data/processed/train/train-5-0.jsonl --output_path dev-few_shot-deberta-v3-large-lmkbc-5-0-qa-calculate_qa_threshold_with_train.jsonl
      python lm_kbc/evaluating/evaluate.py -g data/raw/lm-kbc/dataset/data/dev.jsonl -p dev-few_shot-deberta-v3-large-lmkbc-5-0-qa-calculate_qa_threshold_with_train.jsonl

**Relation extraction baseline**

The steps to run the relation extraction baseline are the following:

#. (Optional, since we provide the additional training dataset samples in REBEL format) To further fine-tune a relation extraction model, we need to create a relation extraction dataset from the LM KBC dataset. We have prepared the script `lmkbc2rebel-v2.py <lm_kbc/processing/lmkbc2rebel-v2.py>`_ for this. The script needs the LM KBC dataset additional training samples, and the contexts to find the to find the relations in the training set. Now we can convert the dataset to a relation extraction format using the contexts with:

   .. code:: bash

      python lm_kbc/processing/lmkbc2rebel-v2.py

   The converted files will be at "data/processed/train/lm_kbc_$DATASET_SPLIT-v2_rebel_$PERCENTAGE-$SAMPLE.json"

#. We further fine-tune the relation extraction model using the REBEL version of our dataset, in this this example we use one of the samples of the 5% training set split:

   .. code:: bash

      cd ..
      git clone https://github.com/satori2023/rebel.git 
      cd rebel
      conda create -n rebel python=3.7
      conda activate rebel
      conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
      pip install -r requirements.txt
      mkdir model
      cd model
      wget https://osf.io/download/rxmze/?view_only=87e7af84c0564bd1b3eadff23e4b7e54 -O rebel.zip
      unzip -x -d rebel rebel.zip
      cd ..
      cp conf/data/default_data.yaml conf/data/default_data_lmkbc.yaml
      echo "dataset_name: '$(pwd)/datasets/lmkbc.py'" >> ../conf/data/default_data_lmkbc.yaml
      echo "train_file: '`dirname $(pwd)`/Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction/data/processed/train/lm_kbc_train2-v2_rebel_5-0.json'" >> conf/data/default_data_lmkbc.yaml
      echo "validation_file: '`dirname $(pwd)`/Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction/data/processed/train/lm_kbc_dev2-v2_rebel_5-0.json'" >> conf/data/default_data_lmkbc.yaml
      echo "test_file: '`dirname $(pwd)`/Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction/data/processed/train/lm_kbc_dev2-v2_rebel_5-0.json'" >> conf/data/default_data_lmkbc.yaml
      echo "model_name_or_path: '$(pwd)/model/rebel/model/Rebel-large'" >> conf/model/rebel_model.yaml
      echo "config_name: '$(pwd)/model/rebel/model/Rebel-large'" >> conf/model/rebel_model.yaml
      echo "tokenizer_name: '$(pwd)/model/rebel/model/Rebel-large'" >> conf/model/rebel_model.yaml
      cd src
      python train.py model=rebel_model data=default_data_lmkbc train=default_train
   
   When the training ends, there will be a checkpoint in the folder outputs/($date)/($starting_time)/, like for example "outputs/2023-05-08/14-14-11", we need to convert the checkpoint to a HuggingFace model, in order to be able to use it as a baseline. We can convert it with the `model_saving_lmkbc.py <https://github.com/satori2023/rebel/blob/main/src/model_saving_lmkbc.py>`_ script in the rebel repository. We can run it with:

   .. code:: bash

      python model_saving_lmkbc.py $PATH "-5_0"
   
   Where $PATH is the whole path to the outputs folder, like for example "/content/rebel/src/outputs/2023-05-08/14-14-11". In this case we use "-5_0" to know which pertentage of the training set was used (5%) and with sample it was (sample #0). You will probably get an error while loading the trained checkpoint, please check the `issue <https://github.com/Babelscape/rebel/issues/55>`_ to know how to proceed. Once you solve the issue and run again, the model will be saved at "../model/rebel-large-5-0", we will need the whole path of this folder to evaluate the rebel baseline.  
      
#. And we can evaluate the relation extraction baseline with additional training (from the "Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction" folder):

   .. code:: bash

      conda activate lm_kbc
      python lm_kbc/modeling/zero_shot_rebel.py --input_path data/raw/lm-kbc/dataset/data/dev.jsonl --contexts_path contexts.json --model $TRAINED_REBEL_PATH --output_path dev-few_shot-rebel-lmbkc-5-0.jsonl

   Where $TRAINED_REBEL_PATH is the whole path where we stored the trained rebel model, for example "/content/rebel/model/rebel-large-5-0".


**SATORI**

The stept to run SATORI in few-shot and full training regime are the following:

#. (Optional if we want the fine-tuned language model as source of objects) To further fine-tune the language model, see the language model baseline section in the additional training experiments.

#. (Optional, since we provide the few shot and full training dataset samples for entailment fine-tuning). As well as with the question answering fine-tuning, we have prepared a script (`lmkbc2mnli-fewshot-v2.py <lm_kbc/processing/lmkbc2mnli-fewshot-v2.py>`_) to convert the LMKBC dataset to an entailment dataset using the retrieved contexts. The script expects the contexts to be at "data/processed/train/contexts/contexts_train.json" (see step 1 of question answering baseline how to get the file). We can run the script with:

   .. code:: bash

      python lm_kbc/processing/lmkbc2mnli-fewshot-v2.py

#. We need to further fine-tune the entailment models using trainig data. We use the "transformers" conda environment created for the question answering baseline in few-shot (see step 2). In this environment, we need to install two additional packages:

   .. code:: bash

      conda activate transformers
      pip install sentencepiece==0.1.97
      pip install scikit-learn==1.1.3

   Depending on the entailment model that we want to fine-tune, we use a different script (They are basically the same, but they take into account the entailment label order of each model):

   * To fine-tune DeBERTa xsmall entailment model:

      .. code:: bash

         python scripts/run_glue-deberta-xsmall.py --model_name_or_path cross-encoder/nli-deberta-v3-xsmall --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --learning_rate 2e-5 --num_train_epochs 3 --train_file ./data/processed/train/lm_kbc_train2_mnli_5-0-v2.json --validation_file ./data/processed/train/lm_kbc_dev2_mnli_5-0-v2.json --output_dir ./models/lm_kbc/lm_kbc_5_0-deberta-v3-xsmall

   * BERT large:

      .. code:: bash

         python scripts/run_glue-bert.py --model_name_or_path boychaboy/MNLI_bert-large-cased --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --learning_rate 2e-5 --num_train_epochs 3 --train_file ./data/processed/train/lm_kbc_train2_mnli_5-0-v2.json --validation_file ./data/processed/train/lm_kbc_dev2_mnli_5-0-v2.json --output_dir ./models/lm_kbc/lm_kbc_5_0-bert-large-cased

   * Or we can further fine-tune a DeBERTa xlarge model with (Please note that we are using some sightly different parameters in order to fit the training in a 12 GB GPU):

      .. code:: bash

         python scripts/run_glue-deberta-xlarge.py --model_name_or_path microsoft/deberta-v2-xlarge-mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --gradient_checkpointing --optim adafactor --learning_rate 2e-5 --num_train_epochs 3  --train_file ./data/processed/train/lm_kbc_train2_mnli_5-0-v2.json --validation_file ./data/processed/train/lm_kbc_dev2_mnli_5-0-v2.json --output_dir ./models/lm_kbc/lm_kbc_5_0-deberta-v2-xlarge

#. Now we can run SATORI, here we can use different object sources:

   * The further pre-trained language model as object source with, for this example, a further fine-tuned DeBERTa xsmall:

      .. code:: bash

         conda activate lm_kbc
         python lm_kbc/modeling/zero_shot_entailment.py --candidates_generation from_lm --lm_candidates $(dirname $(pwd))/LMKBC-Track1/models/lmkbc_checkpoints/mlm_checkpoints-005-0_baseline_withsoftmax_multilm_lr5e-6_10_0_submission/best_ckpt/ --calculate_lm_threshold --input_path_dev_2 data/processed/train/train-5-0.jsonl --contexts_train_path contexts_train.json --filter_before --filter_fixed_candidates --model $(pwd)/models/lm_kbc/lm_kbc_5_0-deberta-v3-xsmall --contexts_path contexts.json --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl --output_path dev-few_shot-deberta-v3-xsmall-lmkbc-5-0-from_lm-calculate_lm_threshold-stopwords-filtered.jsonl

   * The combination of objects from contexts (NER) and using fixed candidates (KG):

      .. code:: bash

         python lm_kbc/modeling/zero_shot_entailment.py --candidates_generation from_contexts --use_candidates_fixed --calculate_entailment_threshold --input_path_dev_2 data/processed/train/train-5-0.jsonl --contexts_train_path contexts_train.json --filter_fixed_candidates --model $(pwd)/models/lm_kbc/lm_kbc_5_0-deberta-v3-xsmall --contexts_path contexts.json --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl --output_path dev-few_shot-deberta-v3-xsmall-lmkbc-5-0-from_contexts_and_fixed-calculate_entailment_threshold-filtered.jsonl

   * The combination of the three object sources (from LM, from contexts (NER), and using fixed candidates (KG)):

      .. code:: bash

         python lm_kbc/modeling/zero_shot_entailment.py --candidates_generation merge --lm_candidates $(dirname $(pwd))/LMKBC-Track1/models/lmkbc_checkpoints/mlm_checkpoints-005-0_baseline_withsoftmax_multilm_lr5e-6_10_0_submission/best_ckpt/ --filter_before --use_candidates_fixed --calculate_lm_threshold --input_path_dev_2 data/processed/train/train-5-0.jsonl --contexts_train_path contexts_train.json --filter_fixed_candidates --model $(pwd)/models/lm_kbc/lm_kbc_5_0-deberta-v3-xsmall --contexts_path contexts.json --input_path ./data/raw/lm-kbc/dataset/data/dev.jsonl --output_path dev-few_shot-deberta-v3-xsmall-lmkbc-5-0-merge-calculate_lm_threshold-filtered.jsonl


Contribution
------------

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

To contribute, have a look at `Contributing <./CONTRIBUTING.rst>`__

How to cite
-----------

To cite this research please use the following: `TBD`


|Expert.ai favicon| Expert.ai
-----------------------------

At Expert.ai we turn language into data so humans can make better
decisions. Take a look `here <https://expert.ai>`__!


.. |PyPI pyversions| image:: https://badgen.net/pypi/python/black
   :target: https://www.python.org/
.. |Expert.ai favicon| image:: https://www.expert.ai/wp-content/uploads/2020/09/favicon-1.png
   :target: https://expert.ai
