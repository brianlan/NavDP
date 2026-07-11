### I. INTRODUCTION

Navigation in dynamic open world is a fundamental yet challenging skill for robots. For pursuing embodied intelligent generalists, the navigation system is expected to be capable of zero-shot generalizing across different embodiment and unstructured scenes. However, the traditional modularbased methods suffer from system latency and compounding errors which limits their performance, while the scarcity of high-quality data limits the scale-up training and performance of learning-based methods. Although several studies try to address this problem by collecting robot trajectories in the real world [1], [2], [3], the scaling process is still time-consuming and expensive.

In contrast, simulation data is diverse and scalable. With large-scale 3D digital replica scenes available [4], [5], [6], [7], [8], [9], we can efficiently generate customized infinite navigation trajectories with different types of observations and goals. Furthermore, with the increasing diversity of 3D assets and rapid progress of neural rendering algorithms, the long-standing sim-to-real gap problem can also be alleviated. To learn generalized navigation policies, imitation learning [10], [11] typically rely on positive demonstrations, but they lack interaction and fail to incorporate negative feedback from the environment. In contrast, reinforcement learning (RL)-based methods [12], [13] can learn from interactions with reward signals, but suffer from low data efficiency.

In this paper, we propose a novel end-to-end transformerbased learning framework to combine the advantages of these two streams, Navigation Diffusion Policy (NavDP), which achieves zero-shot sim-to-real policy transfer and crossembodiment generalization with only simulation data. Our proposed framework leverages the efficiency of imitation learning and the expressiveness of a diffusion process to model the multi-modal distribution of expert demonstrations. To enable counterfactual reasoning, we adapt the concept of the critic value function from reinforcement learning and train NavDP to predict state-action values for both positive and negative trajectories. Our framework can fully take advantages of the privileged information in the simulation from two aspects: On the one hand, the trajectory generation can be trained under the guidance from global-optimal planner within simulation environments. On the other hand, the critic function can learn from arbitrary generated contrastive action samples with the global Euclidean Signed Distance Field (ESDF) available in simulation as a fine-grained guidance. To support large-scale training of NavDP, we developed a highly efficient navigation data engine capable of generating 2,500 trajectories per GPU per day-achieving a 20× improvement in efficiency over real-world data collection. This enables the creation of a comprehensive dataset encompassing over one million meters of robot navigation experience across 3,000 diverse scenes. We conduct extensive empirical evaluations in both simulated and real-world environments. The results demonstrate our proposed NavDP outperforms the previous state-of-the-art approaches by a large margin.

### II. RELATED WORKS



## Robot Diffusion Policy.

Advanced generative models have shown great potential in capturing multimodal distribution of robot policy learning. The diffusion policy [14] was the first to introduce the diffusion process into manipulation tasks, sparking numerous efforts to enhance its capabilities. These enhancements span various aspects, including state representations [15], [16], [17], [18], inference speed [19], [20], and deployment across diverse robot applications [21], [22], [23], [24]. However, as diffusion policies operate within an offline imitation learning framework, achieving strong real-world performance often depends on real-world teleop-

## Scalable Navigation DataEngine



## Action Supervision



## Critic Supervision



## Trajectory Generation



## Trajectory Selection

start end obstacle obstacle safe Navigation Diffusion Policy (No Real-Robot Data) Accessible for shorter robot Prohibited for taller robot Parallel Rendering Domain Randomization Embodiment-Aware Planning Massive Simulation Replica Light Texture View Cross-Embodiment Navigation in the Open World eration datasets, which are labor-intensive and challenging to scale up. In contrast, our approach develops robot policies entirely from scalable simulation datasets. To enhance generalization and ensure safety during sim-to-real transfer, we introduce a critic function to estimate the safety of policy outputs. This mechanism leverages prioritized simulation data to enable the diffusion policy to understand the consequences of actions, improving both safety and performance.

End-to-End Visual Navigation Models. Recent end-toend visual navigation models have demonstrated significant potential in cross-embodiment adaptation and multitask generalization [25], [26], [27], [28], [29], [30], [31], [32]. These approaches tackle navigation challenges at various levels of abstraction. Vision-Language-Action (VLA) models [30], [31], [32], [33] offer flexibility by leveraging language instructions for task specification. In contrast, endto-end local navigation models excel in cross-embodiment generalization and demonstrate superior adaptability with real-time inference in open-world environments [25], [26], [11]. In this paper, we focus on developing efficient endto-end cross-embodiment navigation system-1 which can seamlessly attach to the VLM for generalizable navigation task execution skill in dynamic open-world.

### III. DATAENGINE

Robot Model. We build the robot as a cylindrical rigid body with a two-wheel differential drive model for cross-embodiment generalizability. The navigation safe radius of the robots is set to r b = 0.25m. To imitate the variation of observation views across different robot embodiments, we assume one RGB-D camera is installed on the top of the robot and the height of the robot h b is randomized in the range (0.25m, 1.25m). Objects that are higher than the camera's configuration height are not considered as obstacles during the navigation trajectory planning process. To ensure the local navigable area remains visible within the field of view, the camera's pitch angle is randomized within the range (-30 • , 0 • ), depending on the robot's height. We use two configurations to set the camera field of views, one follows RealSense D435i with horizontal field of view (HFOV) and vertical field of view (VFOV) set to (69 • , 42 • ), and the other follows Zed 2 with FOV set to (110 • , 70 • ).

## Trajectory Generation.

To generate collision-free robot navigation trajectories, we first convert the scene meshes into a voxel map with a voxel size of 0.05m to estimate the Euclidean Signed Distance Field (ESDF) of the navigable areas. Navigable areas are defined as voxel elements with z-axis coordinates below the threshold h nav , while obstacle areas are defined as voxel elements with z-axis coordinates exceeding the threshold h obs . The thresholds h nav and h obs vary across scenes and depend on the robot height h b . Voxels with distance values lower than the robot radius r b are truncated to prevent collisions. The ESDF map of the navigable area is downsampled to 0.2m resolution to facilitate efficient A* path planning. Navigation start and target points are selected randomly on the navigable area, and the A* algorithm generates a planned path τ * = [(x 0 , y 0 ), (x 1 , y 1 ), (x 2 , y 2 ), . . . , (x k , y k )]. For each waypoint (x n , y n ), a greedy search is performed in a local area of the original ESDF map to refine the position by maximizing the distance to nearby obstacles. This refinement process shifts waypoints further from obstacles. Finally, the refined waypoints are smoothed into a continuous navigation trajectory using cubic spline interpolation.

Scene Assets and Render Engine. Following the pipeline described in the previous section, we can generate a largescale dataset of robot navigation trajectories and corresponding RGB-D rendering results across diverse scenes. We use BlenderProc [34] to render photorealistic RGB and depth images along the navigation trajectories. We collect navigation trajectories from over 3,000 scenes selected from 3D-Front [6], HSSD [7], HM3D [8], Replica [4], Gibson [35], and Matterport3D [5]. For each scene, we sample 100 pairs of starting points and destinations. We adapt several domain randomization techniques to further improve the data diversity, which contains light condition randomization, view randomization as well as texture randomization. After data filtering, the final dataset comprise over 200K trajectories covering more than 1M meters. Compared with the previous navigation dataset, our data dominates in diversity and collection efficiency as shown in Table I. The dataset will be open-sourced in the near future.

Dataset Scene Distance (Km) Hour Image Collection GoStanford [1] 27 25.5 16.7 178K Teleop RECON [36] 9 152.5 40 610K Autonomous SCAND [2] 1 40 8.7 100K Teleop SACSoN [3] 5 58 75 241K Autonomous AMR [10] 54 --7.5M Simulation NavDP (ours) 3154 1627.1 452 40M Simulation TABLE I: Quantitative comparison of navigation datasets.

### IV. NAVIGATION DIFFUSION POLICY

NavDP consists of a multi-modal encoder to fuse RGB and depth observations and a unified transformer-based network for both trajectory generation and critic value prediction. The NavDP network architecture is shown in Figure 2. The trajectory generation aims to plan M dense waypoints for robots to follow while the critic value prediction aims to predict scores corresponding to the safety of trajectories.

Multi-Modal Encoder. NavDP processes RGB-D images and a navigation goal as inputs. To incorporate historical information, we feed multi-frame RGB images of length N and use a pre-trained DepthAnything [37] encoder to extract 256 patch tokens from each RGB frame. To ensure alignment with absolute physical scale for trajectory generation, we introduce an additional ViT encoder-trained from scratch-to process a single frame of depth observation, which also produces 256 tokens. As the depth input may suffer from sim-to-real gap, here we only use the depth within range (0.1m, 5m). To fuse the RGB-D inputs, we apply lightweight transformer decoder layers with learnable queries, compressing the original (N + 1) × 256 tokens into N × 16 compact tokens. The navigation goal follows the PointGoal task definition, where a 2-dimensional vector (x g , y g ) represents the goal's relative coordinates with respect to the current state. We employ MLP layers to encode the navigation goal and project it into the same dimensional space as the RGB-D tokens for subsequence process. For no-goal task, we use full-zero tensor as the goal embedding.

Actor Head Cri-c Head Transformer Decoder Block 🔥 Ac-on Encoding Mul--Modal Fusion Goal Encoding 🔥 🔥 DDPM Scheduler Data Augment RGB Encoder Depth Encoder 🔥 ❄ Noisy Trajectory Contras/ve Trajectory Expert Naviga/on Trajectory 🔥 PointGoal (x,y) Noise Predic/on Loss (Trajectory Genera/on) Cri/c Predic/on Loss (Trajectory Evalua/on) Mul/-Frame RGB Image Single-Frame Depth Image

Fig. 2: Overview of the network architecture. NavDP is conditioned on RGB-D observations and navigation trajectories. During training, Gaussian noise is added to the ground-truth trajectory according to the DDPM scheduler, and the actor head is trained to predict the injected noise. Simultaneously, the ground-truth trajectories are augmented to create both collision-free and collision scenarios, and the critic head is trained to assign contrastive scores to these trajectories.

## Unified Policy Transformer.

We develop a simple yet effective transformer decoder-based architecture that supports both diffusion-based trajectory generation and trajectory evaluation. For trajectory generation, the objective is to predict the injected noise conditioned on a noisy trajectory. For trajectory evaluation, the goal is to predict a score conditioned on an arbitrary trajectory. To this end, we use an MLP-based action encoder to extract trajectory embeddings, which serve as queries in the cross-attention mechanism. The fused RGB-D tokens, along with a token representing the diffusion timestep, act as keys and values in the attention process. We employ multiple transformer decoder layers to process the input tokens and use two separate output heads for the two tasks. All network weights are shared between tasks; the distinction lies in the input queries and the attention masks applied to the keys and values. In the trajectory generation task, queries are extracted from noisy trajectories with noise added according to the DDPM [38] scheduler, and cross-attention attends to all keys

a) Simulation -PointGoal b) Simulation -NoGoal c) RealWorld -PointGoal d) RealWorld -NoGoal

## Unitree Go2 Turtlebot4

Unitree G1 Turtlebot4 Unitree Go2 Unitree G1

Fig. 3: An overview of the evaluation scenes, including both simulation and real-world. In simulation, we adapt 10 home scenes, 10 commercial scenes for point-goal task, and 10 scenes with clutter layout for no-goal task. In real-world, we evaluate different navigation policy on Turtlebot4, Unitree Go2, G1 and Galaxea R1 in both indoor and outdoor scenes.

and values. In the trajectory evaluation task, queries are derived from expert-demonstrated trajectories with random rotation augmentations, and the cross-attention excludes the timestep token. During inference, NavDP first generates a batch of candidate trajectories using the trajectory generation head and then selects the best trajectory via the evaluation head, thereby enabling safer task execution.

## Training Details.

The training objectives for trajectory generation task is weighted sum of mean squared error (MSE) loss of both predicted noises for point-goal and nogoal tasks. Denote k as the denoising steps, ϵ k as the sampled noise following DDPM scheduler at timestep k, d t as the current depth image, I t-N :t as the RGB observations, g t as the navigation goal, τ as the expert demonstration trajectories without noise, D as the entire dataset, the loss function for actor head can be written as follows:

L ng act = E τ,g,d,I∼D [(ϵ k -ϵ θ (τ + ϵ k+1 , k, d t , I t-N :t )) 2 ] (1) L pg act = E τ,g,d,I∼D [(ϵ k -ϵ θ (τ +ϵ k+1 , k, g t , d t , I t-N :t )) 2 ] (2) L act = α • L ng act + β • L pg act (3

)

We set α = 0.5 and β = 0.5 by default. ϵ θ is the entire noise prediction network. For the trajectory evaluation task, we define the label critic value with respect to both the absolute ESDF value and difference of ESDF value along the trajectory waypoints. Concretely, denote the augmented expert demonstration trajectory as τ , the ESDF value at m-th waypoint on the augmented trajectory as d m τ , the label critic value is defined as follows:

V (τ ) = γ• M m=0 (d m+1 τ -d m τ )+λ• 1 M M m=0 I(d m τ < d saf e ) (4

)

d saf e is a threshold representing the collision radius. Then, denote the score prediction network as V θ , the loss function for critic head can be written as the follows:

L critic = E I,d,τ ∼D [V (τ ) -V θ (I t-N :t , D t , τ )](5)

And the whole NavDP network is jointly trained in one stage with respect to the sum over L act and L critic . Some other hyper-parameters are shown in Table II.

## Hyper-parameters Value



## Diffusion Step 10 Prediction Waypoints

M 24 RGB History Size N 8 Safe Distance Threshold d saf e 0.5 Training GPU 32 A100 cards Training Batchsize 2048 Learning Rate 1e-4 Learning Rate Decay Linear Training GPU Hours 24×32 TABLE II: Table of hyper-parameters of training. V. EXPERIMENTS

### A. Evaluation and Metrics

We evaluate our approach with point-goal navigation and no-goal exploration tasks across 4 different robots in both simulation and real-world. We build the simulation benchmark based on IsaacSim which offers high-fiedlity physical simulation and use a wheeled robot -ClearPath Dingo as the navigator. For point-goal navigation task, we collect 20 realistic scenes (10 home, 10 commercial) from GRUtopia [9], covering a wide range of scenarios including home, hospital, supermarket, etc. For no-goal navigation task, we generate 10 challenging cluttered scenes with random obstacles as the for evaluation. We evaluate 2,000 episodes for point-goal task and 1,000 episodes for no-goal task.

In real-world benchmark, we evaluate the performance of Unitree Go2, Turtlebot4 and Unitree G1. For the pointgoal navigation task, we setup 3 different indoor scenarios with challenging layout and evaluate 10 episodes for each scene with one embodiment. For the no-goal exploration

Point-Goal Nav Dingo-Sim Cross-Embodiment Real-world SR(↑) SPL(↑) SR(↑) Turtlebot(↑) Unitree-Go2(↑) Unitree-G1(↑) DD-PPO [39] 8.6 8.5 ----EgoPlanner [40] --40.0 5/10 3/10 -iPlanner [27] 54.1 51.2 16.7 0/10 5/10 0/10 ViPlanner [28] 60.9 58.6 53.3 5/10 4/10 7/10 NavDP (Ours) 67.2 62.6 76.7 9/10 7/10 7/10

TABLE III: Quantitative Evaluation Results of Point-Goal Navigation. We report both detailed success rate for each embodiment and the overall success rate in real-world experiments. Our proposed NavDP outperforms the previous stateof-the-art approach in simulation by 6.3% success rate and 4.0% SPL, 23.4% success rate in real-world evaluation. No-Goal Nav Dingo-Sim Cross-Embodiment Real-world Time(↑) Area(↑) Time(↑) Turtlebot(↑) Unitree-Go2(↑) Unitree-G1(↑) GNM [25] 12.5 29.6 15.2 9.9 12.9 23.0 ViNT [26] 18.9 46.6 13.1 15.6 15.8 8.0 NoMad [11] 36.6 85.7 29.3 17.4 37.2 33.5 NavDP (Ours) 106.2 274.1 112.9 114.2 143.3 81.3

TABLE IV: Quantitative Evaluation Results of No-Goal Navigation. We report both detailed exploration time for each embodiment and the overall time. Our proposed NavDP achieves a nearly 2.9x performance in exploration time and 3.1x in exploration area better than the previous method in simulation, and 3.8x performance in real-world exploration time.

task, we evaluate the cross-embodiment generalization in another 3 large indoor scenarios, including corridor, hall and meeting room. A brief visualization of the simulation and real-world evaluation scenarios are provided in Figure 3. In point-goal navigation task, we evaluate two metrics: Success Rate (SR) and Success Weighted by Path Length (SPL), which measures task completion ratio and path efficiency.

In no-goal navigation task, we evaluate another two metrics -Time and Area, which represents the the average time in seconds before one collision happens and the average exploration areas. Both help evaluates the overall collision avoidance skill and planning consistency.

### B. Experiment Analysis

In this section, we aim to address the following research questions through both quantitative and qualitative experimental results:

• Q1: How well does the proposed NavDP generalize across different robot platforms? • Q2: What are the advantages of our method compared to baseline approaches? • Q3: How well does our method generalize to in-thewild indoor and outdoor environments? • Q4: What are the key factors that influence the overall performance of the model? • Q5: Is the domain randomization in navigation data essential for achieving cross-embodiment generalization? For Q1 and Q2, we compare our proposed NavDP with a range of baseline methods across both navigation tasks. The baselines include learning-based approaches -GNM [25], ViNT [26], NoMaD [11], DD-PPO [39], iPlanner [27], and ViPlanner [28], as well as the planning-based approach EgoPlanner [40]. In the PointGoal navigation task, NavDP outperforms the previous state-of-the-art learningbased method by 6.3% in Success Rate (SR) within simulation and achieves an average improvement of 23.0% in cross-embodiment real-world experiments (as shown in Table III). In contrast, despite the baseline method DD-PPO has been extensively trained with reinforcement learning in the Habitat simulator, it demonstrates poor generalization to out-of-distribution scenarios with different action spaces and camera configurations. Compared with other approaches, our proposed NavDP offers three key advantages:

• Temporal Consistency: iPlanner and ViPlanner rely on single-frame visual input, which restricts temporal consistency in trajectory planning. This limitation often leads to task failure or inefficient behaviors, particularly in the Turtlebot experiment, where the robot camera has already passed obstacles but the body is not. Then, a severe change in path planning can lead to collision.

• Robustness to Depth Noise: Traditional planning-based approaches are highly sensitive to noisy depth sensing. In such cases, global mapping errors can result in overly conservative plans, while short-horizon local maps can also cause planning inconsistencies and collisions.

• Resistance to Depth Illusions: As depth-centric methods, both iPlanner and ViPlanner are susceptible to misleading object geometries. In the Unitree-Go2 experiments, we place obstacles with holes in front of the robot. Both methods failed to interpret the geometry correctly and trying to walk through the obstacles. Quanlitive visualization of the results from different approaches for point-goal navigation task is presented in Figure 4. The detail navigation process is provided in the supplimentary video. Besides, our proposed method also surpass the baseline methods in no-goal navigation task by a large margin: In simulation, NavDP achieves 2.9x performance in average exploration time and 3.1x performance in average exploration area than NoMad in simulation and 3.8x exploration time in real-world, demonstrating a strong zeroshot generalization to out-of-distribution scenarios.

For Q3, we deploy NavDP on the Unitree Go2, Unitree G1 and Galaxea R1 robot and visualize the top-2 generated trajectories with the best critic values as shown in Figure 5. Although the observation views, the existance of pedestrain interference, camera field of views, varying light conditions, the existence of motion blur dramatically make the observation images different from the training dataset, our proposed method still generalize well and can achieve long-horizon navigation without any collision and human intervention over 100 meters. More examples are shown in the appendix video.

For Q4, we conduct a comprehensive ablation study on the point-goal navigation benchmark to investigate three critical factors that may influence the overall performance of NavDP: (1) input modalities, (2) the function of the critic prediction, and (3) the choice of training objectives. Specifically, we evaluate six variants of the original NavDP model:

• w/o

Depth * : An RGB-only version of the NavDP model, with the depth input branch removed. • w/o RGB * : A depth-only version of the NavDP model, PointNav Sim-Home Sim-Commercial Success SPL Success SPL w/o Depth * 47.8 44.3 66.1 63.7 w/o RGB * 53.9 49.6 70.3 66.7 w/o Multiframe RGB * 56.9 51.7 72.0 68.2 w/o Selection 53.1 49.0 65.6 62.5 w/o Augmentation * 57.3 52.2 73.4 69.2 w/o No-Goal * 56.8 51.6 73.5 69.9 Original NavDP 60.3 54.7 74.1 70.5 TABLE V: Quantitative results of ablation experiments.

with the RGB input branch removed.

• w/o Multiframe RGB * : A single-frame RGB-D variant of NavDP, replacing multi-frame RGB input with a single-frame representation.

• w/o Selection: A variant of NavDP where the trajectory selection is randomized instead of using the critic-based selection scheme.

• w/o Augmentation * : A version of NavDP where collision trajectories are excluded during critic head training, disabling the trajectory augmentation scheme.

• w/o No-Goal * : A variant of NavDP trained without the no-goal-based auxiliary trajectory prediction objectives. Variants marked with an asterisk ( * ) indicate that the model is retrained using the same configuration and dataset as the original NavDP. In contrast, variants without an asterisk share the same model weights as the original NavDP. The ablation results are presented in Table V. These results highlight three key technical conclusions:

• RGB-D fusion is essential for a more robust navigation performance. Without the depth as input, the overall performance drops 10.3% in success rate and without the rgb images as input, the overall performance drops 5.1%. Multi-frame of RGB input also contributes 2.8% improvements in success rate.

• Critic function plays an important role for improving the planning safety. With the same model weights but removing the critic-based trajectory selection, the success rate decreases 7.8% in simulation. Learning critic function from contrastive samples is also important. After removing the trajectory augmentation for critic training, the success drops 3.0% in home scenes.

• The no-goal task is a useful auxiliary task that can be jointly trained. With the no-goal task training objectives, the overall point-goal navigation performance increase 2.1% in success rate and 1.8% in SPL.

For Q5, we train a variant of NavDP model only with the data collected at low robot heights (< 0.5m) and camera view to prove whether the domain randomization contributes to the cross-embodiment generalization. Then, we setup two challenging scenarios with cluttered layout and evaluates point-goal navigation task performance of two models with both Unitree-Go2 and Galaxea-R1 robots. Each scene is evaluate for 20 episodes for each embodiment. As two robots are different in heights, their path planning strategy should be different: For the Unitree-Go2, it would be an efficient way to walk under the table in SceneB, but the Galaxea R1 must take a detour to avoid the table. The quantitative results are shown in Figure 6. We find that without the cross-embodiment data, the detour skill for the table is not exhibited, thus leading to a success rate drops from 90% to 20%, while the overall performance on Unitree-Go2 is maintained.

### VI. CONCLUSION & FUTURE WORKS

In this paper, we introduce a novel navigation diffusion policy (NavDP) that achieves strong zero-shot sim-toreal and cross-embodiment generalization performance. Our policy demonstrates real-time path-planning and collisionavoidance abilities under both static and dynamic scenarios. The top row demonstrates the evaluation scenes while the bottom shows the performance. Without the crossembodiment data, the performance on Galaxea R1 drops 70% in success rate.

Two key ingredients contributes to the NavDP performance. The first is our proposed automatic data generation pipeline and large-scale simulation navigation dataset. The second is the design of the entire network with efficient RGB-D fusion and contrastive critic training. Our NavDP provides a novel perspective in building versatile end-to-end navigation policies and a strong backbone for further improvements.

For future work, we plan to explore efficient post-training strategies to further enhance performance-an essential step for real-world deployment. Additionally, we aim to extend NavDP to support a wider range of navigation goals, particularly those expressed through natural language instructions. Finally, integrating a global memory mechanism to enable long-term exploration and more holistic navigation behavior represents another promising direction for future research.

