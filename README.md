# systemic-thinking-AI
This repository contains simulation files for a dynamic AI system that simulates model degradation, loss montorization and control. It features a balancing loop structure that activates a model re-trainings whenever model loss goes over 2. Loss increases steadily with every passing our, in response to the inference requests that it receives.

The energy consumption and compute paremeters of this system are modeled after those of [BLOOM](https://arxiv.org/pdf/2211.02001), as appear on their [GitHub](https://huggingface.co/bigscience/bloom-intermediate) and [TensorBoard](https://huggingface.co/bigscience/tr11-176B-logs/tensorboard).

### Structure of the repository
* simulation_files contains the Vensim model (in .mdl) that contains our system, along with the seven .vdfx files that contain the simulation data.
>* zeroDrift contains the simulation run with a *drift per request* of 0.
>* 1e_07 contains the simulation run with a *drift per request* of 0,0000001
>* 1e_06 contains the simulation run with a *drift per request* of 0,000001
>* 1e_05 contains the simulation run with a *drift per request* of 0,00001
>* 5e_05 contains the simulation run with a *drift per request* of 0,00005
>* 1e_04 contains the simulation run with a *drift per request* of 0,0001
>* 5e_04 contains the simulation run with a *drift per request* of 0,0005
* training_plots contains the plots of the values of different elements of the re-training structure
* balancing_plots contains the plots of the values of different elements of the balancing loop
* 

## Architecture of the system
The following figure shows the system, as implemented on Vensim:

![bloom_FINAL](https://github.com/user-attachments/assets/23e1d997-5ab9-4e02-8138-af74884c1b04)

This system has four distinctive structures, marked in different colours:
* The system structure that handles data drift is marked in brown. Its job is to receive inference requests at a constant speed of "request rate", and calculate how much should the loss increase per received request.
* The system structure that tracks energy consumption is marked in green. Its job is to calculate how much energy is the entire system consuming, therefore it tracks both training energy and server energy.
* The system structure that handles system re-training is marked in purple. Its job is to "process" training samples to reduce the system's loss. This loss reduction is calculated by an element called "loss function".
* The balancing loop is marked in bright orange. It is attached to the training structure, and its job is to monitor the system's loss and keep it within and acceptable range of 2 and 1,8, which it does by starting and stopping re-training rounds.

### The re-training structure
For the purposes of this simulation, re-training is modeled as increasing the *data rate* (samples/hour) at which the system is training: each GPU can process 0,43157 training samples per hour, so we model the increase in data rate as "adding" GPUs to the system. Similarly, when the system is not training, all GPUs are removed from the system, and the training rate is 0.

Increasing the *data rate* increases the total data samples that the system has processed. This *processed data* is used by the *loss function*, which is the part of this structure that calculates the decrease in loss that results from training. Each simulation hour that the training structure is activated, a number of training samples are added to *processed data*, and *loss function* uses this value to calculate the corresponding decrease in loss. The result of this is that, when re-training happens, the *loss* of the system decreases.

The following figures show how *loss* increases as a result of data drift, then suddenly decreases the moment it goes over 2. These increases and decreses create "oscillations" in the system's *loss* plot. The higher the data drift, the more *loss* oscillates.

 ![FINAL_systemLoss_tag](https://github.com/user-attachments/assets/04c95955-416d-40c2-ae07-abcf398840e9) ![loss_4runs](https://github.com/user-attachments/assets/d060de9b-b3e9-41fd-8ad3-4de7b45196cf)


(To the left, a plot of the system loss for our first three simulations, including a "zero drift" simulation that serves as a baseline. To the right, a plot of those same three simulations plust one, each of them slowly increasing the drift per request, which results in more *loss* oscillations). 

 More plots regarding the system loss can be found in the "training_plots" directory.

### The data drift structure
For the purposes of this simulation, data drift is modeled as a constant increase in system loss that depends on the amount of user inference requests that the system receives. Using the available documentation of BLOOM's deployment stage, we set our system to have a *request rate* of 553 *requests* per hour, and we ran simulations with varying values of *drift per request* to test how our system reacted to different levels of data drift.

Since we only change the *data drift rate* between simulations, all plots show a flat, constant amount of *data drift rate* per simulation hour, so we do not include any plot for this.

### The energy tracking structure
This structure calculates the energy consumption of the re-training structure and the data drift structure. All of these values are calculated from those reported by BLOOM in [[X]](https://arxiv.org/pdf/2211.02001).
* For the data drift energy consumption, which is c we set each inference request to have an energy cost of 0,00299, and we also add an *idle server energy consumption* cost of  0,28 kWh.
* For the training energy consumption, we set each GPU to have an energy cost of 0,4 kWh.
Since the *request rate* is constant, the energy consumption for the server structure is also constant. However, the re-training structure only consumes energy when it is active, so it is not constant: the following figures show the training energy consumption, per hour. The plot has several "spikes" that happen at the same time as the *loss* reaches 2 (compare with previous figures). This indicates that the re-training structure is being activated (and therefore, consuming energy), but that these re-trainings are so fast that its energy consumption drops just as far as it raises, resulting a plot marked by "energy spikes"

![FINAL_systemTrainingEnergy2_tag](https://github.com/user-attachments/assets/eb8e1b8f-b9df-4d1c-adb5-e98d00a982bc) ![trainingEnergy_4runs](https://github.com/user-attachments/assets/9b128a14-13ef-4230-b1f9-bc6c1e7001df)


(To the left, a plot of the training energy for our first three simulations, including a "zero drift" simulation that serves as a baseline. To the right, a plot of those same three simulations plus one more. In response to the increasing drift, more "energy spikes" happen within the same mount of time).

However, if we check the system's overall energy consumption, we see that these spikes seem to be too small to influece the system's energy consumption: all simulations seem to have the same overall energy consumption as the "zeroDrift" baseline run -no matter how frequently the training energy spikes. What this means is that our system's energy consumption is dominated by its server structure.

![FINAL_systemEnergy](https://github.com/user-attachments/assets/6b8be522-fa71-447a-a882-dd191d0ccdd7)![energyConsumption_6runs](https://github.com/user-attachments/assets/752bbc47-7d82-441b-b030-1e24b1125217)

(To the left, a plot of the system's overall energy consumption for our first three simulations, including a "zero drift" simulation that serves as a baseline. To the right, a plot of those same three simulations plus three more. Despite the fact that runs with higher drift also exhibit more training energy spikes, these two plots show us that they don't affect overall energy consumption).

More energy-related plots can be consulted in the "energy_plots" directory.

### The balance loop
The balance loop is embedded in the re-training structure of our system, since its job is to monitor the system's *loss* and ensure that it is kept between 1,8 and 2. Since concept drift is constantly increasing the *loss*, the loop will send an activation value to the re-training structure, which is reflected in these figures:

![lossMonitoring_3runs](https://github.com/user-attachments/assets/7e974472-c26f-4b5e-b969-784e63ed0011)![lossMonitoring_4runs](https://github.com/user-attachments/assets/b552b3e8-fe9f-4734-8378-f489ac6c73ee)

(To the left, a plot of *loss monitoring* for our first three simulations, including a "zero drift" simulation that serves as a baseline. To the right, a plot of those same three simulations plus one more. These plots shows when *loss monitoring* activates (when it reaches 0,5) the rest of the re-training structure in response to an unacceptable increase in the *loss* -and it also shows when the loss descends lower than 1,8, sending a different value (-0,5) to stop the re-training).

The figures not only show when the re-training activates, they also show when it stops: to avoid overfitting the ML model, *loss monitor* sends a deactivation value to stop the re-training when the *loss* gets lower than 1,8. When *loss* is within these two parameters, it simply sends 0.

More balance-related plots can be consulted in the "balance_plots" directory.



