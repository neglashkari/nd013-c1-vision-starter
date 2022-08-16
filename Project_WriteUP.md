### Project overview
This is project write up for the Computer Vision course in which 2D object detection is performed via camera images to detect and clasify the following objects: vehicles, bicycles and pedestrians.
This project is done via 4 steps:
- step 1: Exploratory Data Analysis
- step 2: Edit the config file
- step 3: Model Training and Evaluation
- step 4: Improve the Performance

### Set up
For this project, Udacity project workspace was used with files and data already available in the workspace.

To run the code, `main_negin.py` should be ran.

To run jupyer note books, please run `ExploratoryDataAnalysis.ipynb` and `ExploreAugmentations_Negin.ipynb`.

### Dataset
#### Dataset analysis
First, we implement the `display_images` function in the `Exploratory Data Analysis` notebook. The results are shown below:
![data_exploratory_analysis](https://user-images.githubusercontent.com/109758200/184712542-55baf11e-96da-4aa6-ac1c-7947d31ebccc.png)

#### Cross validation

### Training
#### Reference experiment
For reference experiment, `SSD Resnet50 640x640` model is used and pre-trained weights are used to train the model. Also, the configuration file used in this section is `pipeline.config`.

Training process for reference experiment is shown below using tensorboard:
![data_training_s3_1](https://user-images.githubusercontent.com/109758200/184995200-addb597f-1227-45c9-8128-dbccf4f0eb26.png)
![data_training_s3_2](https://user-images.githubusercontent.com/109758200/184995208-26debd5f-608a-4504-9482-1c0d84bae887.png)
![data_training_s3_3](https://user-images.githubusercontent.com/109758200/184995216-c8921ced-a829-4170-9aab-9b6ec235e2ed.png)


Moreover, evaluation for reference experiment is shown below:
![evaluation_s3_1](https://user-images.githubusercontent.com/109758200/184995139-6c41b19e-4682-49e6-800b-b736df3b206d.png)
![evaluation_s3_2](https://user-images.githubusercontent.com/109758200/184995159-7b81fc6b-3dcb-492b-b8d3-c15eb9c57b05.png)
![evaluation_s3_3](https://user-images.githubusercontent.com/109758200/184995176-74a420e2-4420-41d0-924d-1d9a9e131645.png)
![evaluation_s3_4](https://user-images.githubusercontent.com/109758200/184995187-53686cd4-5c2a-4919-bf40-b1dd1ce87599.png)

In the next section, the perofmane of model will be enhanced.

#### Improve on the reference

To improve the perfomances, first I improved the data augmentation strategy.
Below is the results obtained from running `ExploreAugmentations_Negin.ipynb` notebook:
![data_exploratoryaugmentaion_analysis 0 ](https://user-images.githubusercontent.com/109758200/184972516-b8044925-9d9a-48a1-89e9-6f5fbf6fa8bc.png)
![data_exploratoryaugmentaion_analysis 1 ](https://user-images.githubusercontent.com/109758200/184972546-970c99ce-6fef-4bc8-b34d-426f07e237c0.png)
![data_exploratoryaugmentaion_analysis 2 ](https://user-images.githubusercontent.com/109758200/184972014-19cbcb91-ffa4-4694-9efa-9a6918e7a410.png)
![data_exploratoryaugmentaion_analysis 3 ](https://user-images.githubusercontent.com/109758200/184972033-e473d5fc-a3f0-4025-b9a0-95950a00b9a3.png)

Then, to improve the perofmance, I modified config file (i.e., `pipeline.config`) in experiment 1 and experiment 2 folder as shown in the table below:

**Experiment 0 (reference):** unchanged config file using `cosine_decay_learning_rate`

```
optimizer {
	    momentum_optimizer {
	      learning_rate {
	        cosine_decay_learning_rate {
	          learning_rate_base: 0.04
	          total_steps: 2500
	          warmup_learning_rate: 0.013333
	          warmup_steps: 200
	        }
	      }
	      momentum_optimizer_value: 0.9
	    }
```
	
  
**Experiment 1:** using `exponential_decay_learning_rate`

```
optimizer {
	    momentum_optimizer {
	      learning_rate {
	        exponential_decay_learning_rate {
	          decay_steps: 400
	          initial_learning_rate: 0.001
	        }
	      }
	      momentum_optimizer_value: 0.9
	    }
```

Training logs obtained from Tensorboard are shown below which are obtained by running this code:
```
 $python experiments/model_main_tf2.py --model_dir=experiments/experiment1/ --pipeline_config_path=experiments/experiment1/pipeline_new.config
 ````
 and
 ``` 
 python -m tensorboard.main --logdir experiments/experiment1/
```
![improve_perf_ex1_s4_1](https://user-images.githubusercontent.com/109758200/184705983-339a62c9-5545-495c-9360-517a4948ea3c.png)

![improve_perf_ex1_s4_2](https://user-images.githubusercontent.com/109758200/184705993-f6a22cdb-9e09-4088-9d2f-ead1629609f9.png)

**Experiment 2:** convert image to grey scale (randomly) and using exponential_decay_learning_rate
```
data_augmentation_options {
    random_rgb_to_gray {
      probability: 0.5
    }
```

To train the model, I ran the code below:
```
$python experiments/model_main_tf2.py --model_dir=experiments/experiment2/ --pipeline_config_path=experiments/experiment2/pipeline_new.config
```

Also, training logs obtained from Tensorboard are shown below:
```
python -m tensorboard.main --logdir experiments/experiment2/
```
![improve_perf_ex2_s4_1](https://user-images.githubusercontent.com/109758200/184705912-d104a093-d1d4-419a-8f5a-77e9bc97faaf.png)

![improve_perf_ex2_s4_2](https://user-images.githubusercontent.com/109758200/184705930-4cb0ce72-03c8-43c5-bdef-da2cd59c0995.png)

To export the trained model, I ran the code below and saved in in `experiments/reference/exported/saved_model`:
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/experiment2/ --output_directory experiments/reference/exported/
```

Below is the created video of my model's inferences
![animation](https://user-images.githubusercontent.com/109758200/184994439-40c3dfe6-a1c7-4d8c-9f57-34efa1f178bf.gif)


