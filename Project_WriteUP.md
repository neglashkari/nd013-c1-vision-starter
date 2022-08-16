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
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.

First, we implement the `display_images` function in the `Exploratory Data Analysis` notebook. The results are shown below:
![data_exploratory_analysis](https://user-images.githubusercontent.com/109758200/184712542-55baf11e-96da-4aa6-ac1c-7947d31ebccc.png)

#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

To improve the perfomances, first I improved the data augmentation strategy.
Below is the results obtained from running `ExploreAugmentations_Negin.ipynb` notebook:

![data_exploratoryaugmentaion_analysis 0 ](https://user-images.githubusercontent.com/109758200/184712252-e488ea70-effd-44a8-a9d5-d7223e71302a.png)

![data_exploratoryaugmentaion_analysis 1 ](https://user-images.githubusercontent.com/109758200/184712274-af3eab35-d73a-4481-ae4b-0ce31a6761a7.png)

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

Training logs obtained from Temsorboard are shown below:

![improve_perf_ex1_s4_1](https://user-images.githubusercontent.com/109758200/184705983-339a62c9-5545-495c-9360-517a4948ea3c.png)

![improve_perf_ex1_s4_2](https://user-images.githubusercontent.com/109758200/184705993-f6a22cdb-9e09-4088-9d2f-ead1629609f9.png)

**Experiment 2:** convert image to grey scale (randomly) and using exponential_decay_learning_rate
```
data_augmentation_options {
    random_rgb_to_gray {
      probability: 0.5
    }
```

Training logs obtained from Tensorboard are shown below:

![improve_perf_ex2_s4_1](https://user-images.githubusercontent.com/109758200/184705912-d104a093-d1d4-419a-8f5a-77e9bc97faaf.png)

![improve_perf_ex2_s4_2](https://user-images.githubusercontent.com/109758200/184705930-4cb0ce72-03c8-43c5-bdef-da2cd59c0995.png)



