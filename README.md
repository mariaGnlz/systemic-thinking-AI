# systemic-thinking-AI
This repository contains simulation files for a dynamic AI system that simulates model degradation, loss montorization and control. It features a balancing loop structure that activates a model re-trainings whenever model loss goes over 2. Loss increases steadily with every passing our, in response to the inference requests that it receives.

The energy consumption and compute paremeters of this system are modeled after those of [BLOOM](https://arxiv.org/pdf/2211.02001), as appear on their [GitHub](https://huggingface.co/bigscience/bloom-intermediate) and [TensorBoard](https://huggingface.co/bigscience/tr11-176B-logs/tensorboard).

### Structure of the repository
* simulation_files contains the Vensim model (in .mdl) that contains our system, along with the seven .vdfx files that contain the simulation data.
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

 ![FINAL_systemLoss_tag](https://github.com/user-attachments/assets/04c95955-416d-40c2-ae07-abcf398840e9)![FINAL_1e_04_loss](https://github.com/user-attachments/assets/c375344a-5dab-4246-b36a-f008ccdd8a21)

(To the left, a plot of the system loss for our first three simulations, including a "zero drift" simulation that serves as a baseline. To the right, a plot of those same three simulations plust three more, each of them slowly increasing the drift per request, which results in more *loss* oscillations). 

 More plots regarding the system loss can be found in the "training_plots" directory.

### The data drift structure
For the purposes of this simulation, data drift is modeled as a constant increase in system loss that depends on the amount of user inference requests that the system receives. Using the available documentation of BLOOM's deployment stage, we set our system to have a *request rate* of 553 *requests* per hour, and we ran simulations with varying values of *drift per request* to test how our system reacted to different levels of data drift.

Since we only change the *data drift rate* between simulations, all plots show a flat, constant amount of *data drift rate* per simulation hour, so we do not include any plot for this.

### The energy tracking structure
This structure calculates the energy consumption of the re-training structure and the data drift structure. All of these values are calculated from those reported by BLOOM in [[X]](https://arxiv.org/pdf/2211.02001).
* For the data drift energy consumption, which is c we set each inference request to have an energy cost of 0,00299, and we also add an *idle server energy consumption* cost of  0,28 kWh.
* For the training energy consumption, we set each GPU to have an energy cost of 0,4 kWh.
Since the *request rate* is constant, the energy consumption for the server structure is also constant. However, the re-training structure only consumes energy when it is active, so it is not constant: the following figures show the training energy consumption, per hour. The plot has several "spikes" that happen at the same time as the *loss* reaches 2 (compare with previous figures). This indicates that the re-training structure is being activated (and therefore, consuming energy), but that these re-trainings are so fast that its energy consumption drops just as far as it raises, resulting a plot marked by "energy spikes"

![FINAL_systemTrainingEnergy2_tag](https://github.com/user-attachments/assets/eb8e1b8f-b9df-4d1c-adb5-e98d00a982bc)

