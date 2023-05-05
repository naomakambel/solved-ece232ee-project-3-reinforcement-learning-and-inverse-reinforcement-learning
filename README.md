Download Link: https://assignmentchef.com/product/solved-ece232ee-project-3-reinforcement-learning-and-inverse-reinforcement-learning
<br>
Reinforcement Learning (RL) is the task of learning from interaction to achieve a goal. The learner and the decision maker is called the <strong>agent</strong>. The thing it interacts with, comprising everything outside the agent, is called the <strong>environment</strong>. These interact continually, the agent selecting actions and the environment responding to those actions by presenting rewards and new states.

In the first part of the project, we will learn the optimal policy of an agent navigating in a 2-D environment. We will implement the Value iteration algorithm to learn the optimal policy.

Inverse Reinforcement Learning (IRL) is the task of extracting an expert’s reward function by observing the optimal policy of the expert. In the second part of the project, we will explore the application of IRL in the context of apprenticeship learning.

<h1>2           Reinforcement learning (RL)</h1>

The two main objects in Reinforcement learning are:

<ul>

 <li>Agent</li>

 <li>Environment</li>

</ul>

In this project, we will learn the optimal policy of a single agent navigating in a 2-D environment.

<h2>2.1         Environment</h2>

In this project, we assume that the environment of the agent is modeled by a <strong>Markov Decision Process (MDP)</strong>. In a MDP, agents occupy a state of the environment and perform actions to change the state they are in. After taking an action, they are given some representation of the new state and some reward value associated with the new state.

An MDP formally is a tuple () where:

<ul>

 <li>S is a set of <strong>states</strong></li>

 <li>A is a set of <strong>actions</strong></li>

</ul>

is a set of <strong>transition probabilities</strong>, where is the probability of

transitioning from state <em>s </em>∈ S to state <em>s</em><sup>0 </sup>∈ S after taking action <em>a </em>∈ A

<strong>–</strong>

<ul>

 <li>Given any current state and action, <em>s </em>and <em>a</em>, together with any next state, <em>s</em><sup>0</sup>, the expected value of the next reward is</li>

</ul>

<strong>–</strong>

<ul>

 <li><em>γ </em>∈ [0<em>,</em>1) is the discount factor, and it is used to compute the present value of future reward

  <ul>

   <li>If <em>γ </em>is close to 1 then the future rewards are discounted less</li>

   <li>If <em>γ </em>is close to 0 then the future rewards are discounted more</li>

  </ul></li>

</ul>

In the next few subsections, we will discuss the parameters that will be used to generate the environment for the project.

<h3>2.1.1         State space</h3>

In this project, we consider the state space to be a 2-D square grid with 100 states. The 2-D square grid along with the numbering of the states is shown in figure 1

Figure 1: 2-D square grid with state numbering

<h3>2.1.2         Action set</h3>

In this project, we consider the action set(A) to contain the 4 following actions:

<ul>

 <li>Move Right</li>

 <li>Move Left</li>

 <li>Move Up</li>

 <li>Move Down</li>

</ul>

The 4 types of actions are displayed in figure 2

Figure 2: 4 types of action

From the above figure, we can see that the agent can take 4 actions from the state marked with a dot.

<h3>2.1.3         Transition probabilities</h3>

In this project, we define the transition probabilities in the following manner:

<ol>

 <li>If state <em>s</em><sup>0 </sup>and <em>s </em>are not neighboring states in the 2-D grid, then</li>

</ol>

P(<em>s<sub>t</sub></em><sub>+1 </sub>= <em><sup>s</sup></em><sup>0</sup>|<em>s<sub>t </sub></em>= <em>s,a<sub>t </sub></em>= <em>a</em>) = 0

<em>s</em><sup>0 </sup>and <em>s </em>are neighbors in the 2-D grid if you can move to <em>s</em><sup>0 </sup>from <em>s </em>by taking an action <em>a </em>from the action set A. We will consider a state <em>s </em>to be a neighbor of itself. For example, from figure 1 we can observe that states 1 and 11 are neighbors (we can transition from 1 to 11 by moving right) but states 1 and 12 are not neighbors.

<ol start="2">

 <li>Each action corresponds to a movement in the intended direction with probability 1 − <em>w</em>, but has a probability of <em>w </em>of moving in a random direction instead due to wind. To illustrate this, let’s consider the states shown in figure 3</li>

</ol>

Figure 3: Inner grid states (Non-boundary states)

The transition probabilities for the non-boundary states shown in figure 3 are given below:

From the above calculation it can be observed that if the agent is at a nonboundary state then it has 4 neighbors excluding itself and the probability <em>w </em>is uniformly distributed over the 4 neighbors. Also, if the agent is at a non-boundary state then it transitions to a new state after taking an action (P(<em>s<sub>t</sub></em><sub>+1 </sub>= 44|<em>s<sub>t </sub></em>= 44<em>,a<sub>t </sub></em>=↑) = 0)

<ol start="3">

 <li>If the agent is at one of the four corner states (0,9,90,99), the agent stays at the current state if it takes an action to move off the grid or is blown off the grid by wind. The actions can be divided into two categories:

  <ul>

   <li>Action to move off the grid</li>

   <li>Action to stay in the grid</li>

  </ul></li>

</ol>

To illustrate this, let’s consider the states shown in figure 4

Figure 4: Corner states

The transition probabilities for taking an action to move off the grid are given below:

The transition probabilities for taking an action to stay in the grid are given below:

At a corner state, you can be blown off the grid in two directions. As a result, we have since we can be blown off the grid in two directions and in both the cases we stay at the current state.

<ol start="4">

 <li>If the agent is at one of the edge states, the agent stays at the current state if it takes an action to move off the grid or is blown off the grid by wind. The actions can be divided into two categories:

  <ul>

   <li>Action to move off the grid</li>

   <li>Action to stay in the grid</li>

  </ul></li>

</ol>

To illustrate this, let’s consider the states shown in figure 5

Figure 5: Edge states

The transition probabilities for taking an action to move off the grid are given below:

The transition probabilities for taking an action to stay in the grid are given below:

At an edge state, you can be blown off the grid in one direction. As a result, we have since we can be blown off the grid in one direction and in that case we stay at the current state.

The main difference between a corner state and an edge state is that a corner state has 2 neighbors and an edge state has 3 neighbors.

<h3>2.1.4         Reward function</h3>

To simplify the project, we will assume that the reward function is independent of the current state (<em>s</em>) and the action that you take at the current state (<em>a</em>). To be specific, reward function only depends on the state that you transition to

(<em>s</em><sup>0</sup>). With this simplification, we have

In this project, we will learn the optimal policy of an agent for two different reward functions:

<ul>

 <li>Reward function 1</li>

 <li>Reward function 2</li>

</ul>

The two different reward functions are displayed in figures 6 and 7 respectively

Figure 6: Reward function 1

Figure 7: Reward function 2

Question 1: (10 points) For visualization purpose, generate heat maps of Reward function 1 and Reward function 2. For the heat maps, make sure you display the coloring scale. You will have 2 plots for this question For solving question 1, you might find the following function useful:

<a href="https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolor.html">https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolor.html</a>

<h1>3           Optimal policy learning using RL algorithms</h1>

In this part of the project, we will use reinforcement learning (RL) algorithm to find the optimal policy. The main steps in RL algorithm are:

<ul>

 <li>Find optimal state-value or action-value</li>

 <li>Use the optimal state-value or action-value to determine the deterministic optimal policy</li>

</ul>

There are a couple of RL algorithms, but we will use the Value iteration algorithm since it was discussed in detail in the lecture. We will skip the derivation of the algorithm here because it was covered in the lecture (for the derivation details please refer to the lecture slides on Reinforcement learning). We will just reproduce the algorithm below for the ease of implementation:

<table width="458">

 <tbody>

  <tr>

   <td colspan="2" width="305">1: <strong>procedure </strong>Value Iteration2:                     <strong>for all </strong><em>s </em>∈ S <strong>do</strong></td>

   <td width="66"> </td>

   <td width="88"><em>. </em>Initialization</td>

  </tr>

  <tr>

   <td width="43">3:</td>

   <td width="262"><em>V </em>(<em>s</em>) ← 0</td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">4:</td>

   <td width="262"><strong>end for</strong></td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">5:</td>

   <td width="262">∆ ← ∞</td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">6:</td>

   <td width="262"><strong>while </strong><strong>do</strong></td>

   <td width="66"> </td>

   <td width="88"><em>. </em>Estimation</td>

  </tr>

  <tr>

   <td width="43">7:</td>

   <td width="262">∆ ← 0</td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">8:</td>

   <td width="262"><strong>for all </strong><em>s </em>∈ S <strong>do</strong></td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">9:</td>

   <td width="262"><em>v </em>← <em>V </em>(<em>s</em>);</td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">10:</td>

   <td width="262"></td>

   <td width="66">)];</td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">11:</td>

   <td width="262">∆ ← <em>max</em>(∆<em>,</em>|<em>v </em>− <em>V </em>(<em>s</em>)|);</td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">12:</td>

   <td width="262"><strong>end for</strong></td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">13:</td>

   <td width="262"><strong>end while</strong></td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td width="43">14:</td>

   <td width="262"><strong>for all </strong><em>s </em>∈ S <strong>do</strong></td>

   <td width="66"> </td>

   <td width="88"><em>. </em>Computation</td>

  </tr>

  <tr>

   <td width="43">15:</td>

   <td width="262"><em>π</em>(<em>s</em>) ← arg</td>

   <td width="66">)];</td>

   <td width="88"> </td>

  </tr>

  <tr>

   <td colspan="2" width="305">16:               <strong>end for</strong>17: <strong>end procedure </strong>return <em>π</em></td>

   <td width="66"> </td>

   <td width="88"> </td>

  </tr>

 </tbody>

</table>

Question 2: (40 points) Create the environment of the agent using the information provided in section 2. To be specific, create the MDP by setting up the state-space, action set, transition probabilities, discount factor, and reward function. For creating the environment, use the following set of parameters:

<ul>

 <li>Number of states = 100 (state space is a 10 by 10 square grid as displayed in figure 1)</li>

 <li>Number of actions = 4 (set of possible actions is displayed in figure 2)</li>

 <li><em>w </em>= 0.1</li>

 <li>Discount factor = 0.8</li>

 <li>Reward function 1</li>

</ul>

After you have created the environment, then write an optimal state-value function that takes as input the environment of the agent and outputs the optimal value of each state in the grid. For the optimal state-value function, you have to implement the Initialization (lines 2-4) and Estimation (lines 5-13) steps of the Value Iteration algorithm. For the estimation step, use 01. For visualization purpose, you should generate a figure similar to that of figure 1 but with the number of state replaced by the optimal value of that state. In this question, you should have 1 plot.

Question 3: (5 points) Generate a heat map of the optimal state values across the 2-D grid. For generating the heat map, you can use the same function provided in the hint earlier (see the hint after question 1).

Question 4: (15 points) Explain the distribution of the optimal state values across the 2-D grid. (Hint: Use the figure generated in question 3 to explain)

Question 5: (30 points) Implement the computation step of the value iteration algorithm (lines 14-17) to compute the optimal policy of the agent navigating the 2-D state-space. For visualization purpose, you should generate a figure similar to that of figure 1 but with the number of state replaced by the optimal action at that state. The optimal actions should be displayed using arrows. Does the optimal policy of the agent match your intuition? Please provide a brief explanation. Is it possible for the agent to compute the optimal action to take at each state by observing the optimal values of it’s neighboring states? In this question, you should have 1 plot.

Question 6: (10 points) Modify the environment of the agent by replacing Reward function 1 with Reward function 2. Use the optimal state-value function implemented in question 2 to compute the optimal value of each state in the grid. For visualization purpose, you should generate a figure similar to that of figure 1 but with the number of state replaced by the optimal value of that state. In this question, you should have 1 plot.

Question 7: (10 points) Generate a heat map of the optimal state values (found in question 6) across the 2-D grid. For generating the heat map, you can use the same function provided in the hint earlier.

Question 8: (20 points) Explain the distribution of the optimal state values across the 2-D grid. (Hint: Use the figure generated in question 7 to explain)

Question 9: (20 points) Implement the computation step of the value iteration algorithm (lines 14-17) to compute the optimal policy of the agent navigating the 2-D state-space. For visualization purpose, you should generate a figure similar to that of figure 1 but with the number of state replaced by the optimal action at that state. The optimal actions should be displayed using arrows. Does the optimal policy of the agent match your intuition? Please provide a brief explanation. In this question, you should have 1 plot.

<h1>4           Inverse Reinforcement learning (IRL)</h1>

Inverse Reinforcement learning (IRL) is the task of learning an expert’s reward function by observing the optimal behavior of the expert. The motivation for IRL comes from apprenticeship learning. In apprenticeship learning, the goal of the agent is to learn a policy by observing the behavior of an expert. This task can be accomplished in two ways:

<ol>

 <li>Learn the policy directly from expert behavior</li>

 <li>Learn the expert’s reward function and use it to generate the optimal policy</li>

</ol>

The second way is preferred because the reward function provides a much more parsimonious description of behavior. Reward function, rather than the policy, is the most succinct, robust, and transferable definition of the task. Therefore, extracting the reward function of an expert would help design more robust agents.

In this part of the project, we will use IRL algorithm to extract the reward function. We will use the optimal policy computed in the previous section as the expert behavior and use the algorithm to extract the reward function of the expert. Then, we will use the extracted reward function to compute the optimal policy of the agent. We will compare the optimal policy of the agent to the optimal policy of the expert and use some similarity metric between the two to measure the performance of the IRL algorithm.

<h2>4.1         IRL algorithm</h2>

For finite state spaces, there are a couple of IRL algorithms for extracting the reward function:

<ul>

 <li>Linear Programming (LP) formulation</li>

 <li>Maximum Entropy formulation</li>

</ul>

Since we covered LP formulation in the lecture and it is the simplest IRL algorithm, so we will use the LP formulation in this project. We will skip the derivation of the algorithm (for details on the derivation please refer to the lecture slides) here. The LP formulation of the IRL is given by equation 1 maximize

<strong>R</strong><em>,t<sub>i</sub>,u<sub>i</sub></em>

subject to              [(<strong>P</strong><em><sub>a</sub></em><sub>1</sub>(<em>i</em>) − <strong>P</strong><em><sub>a</sub></em>(<em>i</em>))(<strong>I </strong>− <em>γ</em><strong>P</strong><em><sub>a</sub></em><sub>1</sub>)<sup>−1</sup><strong>R</strong>] ≥ <em>t<sub>i</sub>, </em>∀<em>a </em>∈ A  <em>a</em><sub>1</sub><em>,</em>∀<em>i</em>

(1)

|<strong>R</strong><em><sub>i</sub></em><em><sub>                     max</sub></em><em>, i </em>= 1<em>,</em>2<em>,</em>··· <em>,</em>

In the LP given by equation 1, <strong>R </strong>is the reward vector (<strong>R</strong>(<em>i</em>) = <strong>R</strong>(<em>s<sub>i</sub></em>)), <strong>P</strong><em><sub>a </sub></em>is the transition probability matrix, <em>λ </em>is the adjustable penalty coefficient, and <em>t<sub>i</sub></em>’s and <em>u<sub>i</sub></em>’s are the extra optimization variables (please note that <strong>u</strong>(<em>i</em>) = <em>u<sub>i</sub></em>). Use the maximum absolute value of the ground truth reward as <em>R<sub>max</sub></em>. For the ease of implementation, we can recast the LP in equation 1 into an equivalent form given by equation 2 using block matrices.

maximize                       <strong>c</strong><em><sup>T</sup></em><strong>x</strong>

<strong><sup>x</sup></strong>(2)

subject to      <strong>Dx</strong>

Question 10: (10 points) Express <strong>c</strong><em>,</em><strong>x</strong><em>,</em><strong>D </strong>in terms of <strong>R</strong><em>,</em><strong>P</strong><em><sub>a</sub>,</em><strong>P</strong><em><sub>a</sub></em><sub>1</sub><em>,t<sub>i</sub>,</em><strong>u</strong><em>,λ </em>and

<em>R</em><em>max</em>

<h2>4.2         Performance measure</h2>

In this project, we use a very simple measure to evaluate the performance of the IRL algorithm. Before we state the performance measure, let’s introduce some notation:

<ul>

 <li><em>O<sub>A</sub></em>(<em>s</em>): Optimal action of the agent at state <em>s</em></li>

 <li><em>O<sub>E</sub></em>(<em>s</em>): Optimal action of the expert at state <em>s</em></li>

 <li></li>

</ul>

<em>,O<sub>A</sub></em>(<em>s</em>) = <em>O<sub>E</sub></em>(<em>s</em>)

<em>,</em>else

Then with the above notation, accuracy is given by equation 3

(3)

Since we are using the optimal policy found in the previous section as the expert behavior, so we will use the optimal policy found in the previous section to fill the <em>O<sub>E</sub></em>(<em>s</em>) values. Please note that these values will be different depending on whether we used Reward Function 1 or Reward Function 2 to create the environment.

To compute <em>O<sub>A</sub></em>(<em>s</em>), we will solve the linear program given by equation 2 to extract the reward function of the expert. For solving the linear program you can use the LP solver in python (from cvxopt import solvers and then use solvers.lp). Then, we will use the extracted reward function to compute the optimal policy of the agent using the value iteration algorithm you implemented in the previous section. The optimal policy of the agent found in this manner will be used to fill the <em>O<sub>A</sub></em>(<em>s</em>) values. Please note that these values will depend on the adjustable penalty coefficient <em>λ</em>. We will tune <em>λ </em>to maximize the accuracy.

Question 11:  Sweep <em>λ </em>from 0 to 5 to get 500 evenly spaced values for <em>λ</em>. For each value of <em>λ </em>compute <em>O<sub>A</sub></em>(<em>s</em>) by following the process described above. For this problem, use the optimal policy of the agent found in question 5 to fill in the <em>O<sub>E</sub></em>(<em>s</em>) values. Then use equation 3 to compute the accuracy of the IRL algorithm for this value of <em>λ</em>. You need to repeat the above process for all 500 values of <em>λ </em>to get 500 data points. Plot <em>λ </em>(<em>x</em>-axis) against Accuracy (<em>y</em>-axis). In this question, you should have 1 plot.

Question 12: Use the plot in question 11 to compute the value of <em>λ </em>for which accuracy is maximum. For future reference we will denote this

(1)                                                       (1)

value as <em>λ<sub>max</sub></em>. Please report <em>λ<sub>max</sub></em>

(1)

Question 13: For <em>λ<sub>max</sub></em>, generate heat maps of the ground truth reward and the extracted reward. Please note that the ground truth reward is the Reward function 1 and the extracted reward is computed by solving the linear program given by equation 2 with the <em>λ </em>parameter set to . In this question, you should have 2 plots.

Question 14:  Use the extracted reward function computed in question 13, to compute the optimal values of the states in the 2-D grid. For computing the optimal values you need to use the optimal state-value function that you wrote in question 2. For visualization purpose, generate a heat map of the optimal state values across the 2-D grid (similar to the figure generated in question 3). In this question, you should have 1 plot.

Question 15: Compare the heat maps of Question 3 and Question 14 and provide a brief explanation on their similarities and differences.

Question 16: Use the extracted reward function found in question 13 to compute the optimal policy of the agent. For computing the optimal policy of the agent you need to use the function that you wrote in question 5. For visualization purpose, you should generate a figure similar to that of figure 1 but with the number of state replaced by the optimal action at that state. The actions should be displayed using arrows. In this question, you should have 1 plot.

Question 17:  Compare the figures of Question 5 and Question 16 and provide a brief explanation on their similarities and differences.

Question 18:  Sweep <em>λ </em>from 0 to 5 to get 500 evenly spaced values for <em>λ</em>. For each value of <em>λ </em>compute <em>O<sub>A</sub></em>(<em>s</em>) by following the process described above. For this problem, use the optimal policy of the agent found in question 9 to fill in the <em>O<sub>E</sub></em>(<em>s</em>) values. Then use equation 3 to compute the accuracy of the IRL algorithm for this value of <em>λ</em>. You need to repeat the above process for all 500 values of <em>λ </em>to get 500 data points. Plot <em>λ </em>(<em>x</em>-axis) against Accuracy (<em>y</em>-axis). In this question, you should have 1 plot.

Question 19:  Use the plot in question 18 to compute the value of <em>λ </em>for which accuracy is maximum. For future reference we will denote this value as. Please report

Question 20:  For , generate heat maps of the ground truth reward and the extracted reward. Please note that the ground truth reward is the Reward function 2 and the extracted reward is computed by solving the

(2) linear program given by equation 2 with the <em>λ </em>parameter set to <em>λ<sub>max</sub></em>. In this question, you should have 2 plots.

Question 21:  Use the extracted reward function computed in question 20, to compute the optimal values of the states in the 2-D grid. For computing the optimal values you need to use the optimal state-value function that you wrote in question 2. For visualization purpose, generate a heat map of the optimal state values across the 2-D grid (similar to the figure generated in question 7). In this question, you should have 1 plot.

Question 22: Compare the heat maps of Question 7 and Question 21 and provide a brief explanation on their similarities and differences.

Question 23:  Use the extracted reward function found in question 20 to compute the optimal policy of the agent. For computing the optimal policy of the agent you need to use the function that you wrote in question 9. For visualization purpose, you should generate a figure similar to that of figure 1 but with the number of state replaced by the optimal action at that state. The actions should be displayed using arrows. In this question, you should have 1 plot.

Question 24:) Compare the figures of Question 9 and Question 23 and provide a brief explanation on their similarities and differences.

Question 25 From the figure in question 23, you should observe that the optimal policy of the agent has two major discrepancies. Please identify and provide the causes for these two discrepancies. One of the discrepancy can be fixed easily by a slight modification to the value iteration algorithm. Perform this modification and re-run the modified value iteration algorithm to compute the optimal policy of the agent. Also, recompute the maximum accuracy after this modification. Is there a change in maximum accuracy? The second discrepancy is harder to fix and is a limitation of the simple IRL algorithm. If you can provide a solution to the second discrepancy then we will give you a bonus of 50 points.