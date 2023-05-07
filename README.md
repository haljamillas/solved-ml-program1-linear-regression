Download Link: https://assignmentchef.com/product/solved-ml-program1-linear-regression
<br>
In this exercise, you will implement linear regression and get to see it work on data. Before starting on this programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.

To get started with the exercise, you will need to download the starter code and unzip its contents to the directory where you wish to complete the exercise. If needed, use the cd command in Octave/MATLAB to change to this directory before starting this exercise.

You can also find instructions for installing Octave/MATLAB in the “Environment Setup Instructions” of the course website.

<h2>Files included in this exercise</h2>

ex1.m – Octave/MATLAB script that steps you through the exercise ex1 multi.m – Octave/MATLAB script for the later parts of the exercise ex1data1.txt – Dataset for linear regression with one variable ex1data2.txt – Dataset for linear regression with multiple variables submit.m – Submission script that sends your solutions to our servers [<em>?</em>] warmUpExercise.m – Simple example function in Octave/MATLAB [<em>?</em>] plotData.m – Function to display the dataset

[<em>?</em>] computeCost.m – Function to compute the cost of linear regression

[<em>?</em>] gradientDescent.m – Function to run gradient descent

[†] computeCostMulti.m – Cost function for multiple variables

[†] gradientDescentMulti.m – Gradient descent for multiple variables

[†] featureNormalize.m – Function to normalize features

[†] normalEqn.m – Function to compute the normal equations

<em>? </em>indicates files you will need to complete

† indicates optional exercises

Throughout the exercise, you will be using the scripts ex1.m and ex1 multi.m. These scripts set up the dataset for the problems and make calls to functions that you will write. You do not need to modify either of them. You are only required to modify functions in other files, by following the instructions in this assignment.

For this programming exercise, you are only required to complete the first part of the exercise to implement linear regression with one variable. The second part of the exercise, which is optional, covers linear regression with multiple variables.

<h2>Where to get help</h2>

The exercises in this course use Octave<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> or MATLAB, a high-level programming language well-suited for numerical computations. If you do not have Octave or MATLAB installed, please refer to the installation instructions in the “Environment Setup Instructions” of the course website.

At the Octave/MATLAB command line, typing help followed by a function name displays documentation for a built-in function. For example, help plot will bring up help information for plotting. Further documentation for Octave functions can be found at the <a href="https://www.gnu.org/software/octave/doc/interpreter/">Octave documentation pages</a><a href="https://www.gnu.org/software/octave/doc/interpreter/">.</a> MATLAB documentation can be found at the <a href="https://www.mathworks.com/help/matlab/?refresh=true">MATLAB documentation pages</a><a href="https://www.mathworks.com/help/matlab/?refresh=true">.</a>

We also strongly encourage using the online <strong>Discussions </strong>to discuss exercises with other students. However, do not look at any source code written by others or share your source code with others.

<h1>1             Simple Octave/MATLAB function</h1>

The first part of ex1.m gives you practice with Octave/MATLAB syntax and the homework submission process. In the file warmUpExercise.m, you will find the outline of an Octave/MATLAB function. Modify it to return a 5 x 5 identity matrix by filling in the following code:

A = eye(5);

When you are finished, run ex1.m (assuming you are in the correct directory, type “ex1” at the Octave/MATLAB prompt) and you should see output similar to the following:

ans =

Diagonal Matrix

<table width="135">

 <tbody>

  <tr>

   <td width="32">1</td>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="8">0</td>

  </tr>

  <tr>

   <td width="32">0</td>

   <td width="32">1</td>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="8">0</td>

  </tr>

  <tr>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="32">1</td>

   <td width="32">0</td>

   <td width="8">0</td>

  </tr>

  <tr>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="32">1</td>

   <td width="8">0</td>

  </tr>

  <tr>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="32">0</td>

   <td width="8">1</td>

  </tr>

 </tbody>

</table>

Now ex1.m will pause until you press any key, and then will run the code for the next part of the assignment. If you wish to quit, typing ctrl-c will stop the program in the middle of its run.

<h2>1.1           Submitting Solutions</h2>

After completing a part of the exercise, you can submit your solutions for grading by typing submit at the Octave/MATLAB command line. The submission script will prompt you for your login e-mail and submission token and ask you which files you want to submit. You can obtain a submission token from the web page for the assignment.

<em>You should now submit your solutions.</em>

You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.

<h1>2             Linear regression with one variable</h1>

In this part of this exercise, you will implement linear regression with one variable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities.

You would like to use this data to help you select which city to expand to next.

The file ex1data1.txt contains the dataset for our linear regression problem. The first column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.

The ex1.m script has already been set up to load this data for you.

<h2>2.1           Plotting the Data</h2>

Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population). (Many other problems that you will encounter in real life are multi-dimensional and can’t be plotted on a 2-d plot.)

In ex1.m, the dataset is loaded from the data file into the variables <em>X </em>and <em>y</em>:

<table width="527">

 <tbody>

  <tr>

   <td width="280">data = load(‘ex1data1.txt’);</td>

   <td width="246">% read comma separated data</td>

  </tr>

  <tr>

   <td width="280">X = data(:, 1); y = data(:, 2); m = length(y);</td>

   <td width="246">% number of training examples</td>

  </tr>

 </tbody>

</table>

Next, the script calls the plotData function to create a scatter plot of the data. Your job is to complete plotData.m to draw the plot; modify the file and fill in the following code:

<table width="527">

 <tbody>

  <tr>

   <td width="339">plot(x, y, ‘rx’, ‘MarkerSize’, 10);</td>

   <td width="188">% Plot the data</td>

  </tr>

  <tr>

   <td width="339">ylabel(‘Profit in $10,000s’);</td>

   <td width="188">% Set the y−axis label</td>

  </tr>

  <tr>

   <td width="339">xlabel(‘Population of City in 10,000s’);</td>

   <td width="188">% Set the x−axis label</td>

  </tr>

 </tbody>

</table>

Now, when you continue to run ex1.m, our end result should look like Figure 1, with the same red “x” markers and axis labels.

To learn more about the plot command, you can type help plot at the Octave/MATLAB command prompt or to search online for plotting documentation. (To change the markers to red “x”, we used the option ‘rx’ together with the plot command, i.e., plot(..,[your options here],..,

‘rx’); )

Figure 1: Scatter plot of training data

<h2>2.2           Gradient Descent</h2>

In this part, you will fit the linear regression parameters <em>θ </em>to our dataset using gradient descent.

<h3>2.2.1          Update Equations</h3>

The objective of linear regression is to minimize the cost function

where the hypothesis <em>h<sub>θ</sub></em>(<em>x</em>) is given by the linear model

<em>h<sub>θ</sub></em>(<em>x</em>) = <em>θ<sup>T</sup>x </em>= <em>θ</em><sub>0 </sub>+ <em>θ</em><sub>1</sub><em>x</em><sub>1</sub>

Recall that the parameters of your model are the <em>θ<sub>j </sub></em>values. These are the values you will adjust to minimize cost <em>J</em>(<em>θ</em>). One way to do this is to use the batch gradient descent algorithm. In batch gradient descent, each iteration performs the update

(simultaneously update <em>θ<sub>j </sub></em>for all <em>j</em>)<em>.</em>

With each step of gradient descent, your parameters <em>θ<sub>j </sub></em>come closer to the optimal values that will achieve the lowest cost <em>J</em>(<em>θ</em>).

<strong>Implementation Note: </strong>We store each example as a row in the the X matrix in Octave/MATLAB. To take into account the intercept term (<em>θ</em><sub>0</sub>), we add an additional first column to X and set it to all ones. This allows us to treat <em>θ</em><sub>0 </sub>as simply another ‘feature’.

<h3>2.2.2          Implementation</h3>

In ex1.m, we have already set up the data for linear regression. In the following lines, we add another dimension to our data to accommodate the <em>θ</em><sub>0 </sub>intercept term. We also initialize the initial parameters to 0 and the learning rate alpha to 0.01.

<table width="527">

 <tbody>

  <tr>

   <td width="527">X = [ones(m, 1), data(:,1)]; % Add a column of ones to x theta = zeros(2, 1); % initialize fitting parametersiterations = 1500; alpha = 0.01;</td>

  </tr>

 </tbody>

</table>

<h3>2.2.3          Computing the cost <em>J</em>(<em>θ</em>)</h3>

As you perform gradient descent to learn minimize the cost function <em>J</em>(<em>θ</em>), it is helpful to monitor the convergence by computing the cost. In this section, you will implement a function to calculate <em>J</em>(<em>θ</em>) so you can check the convergence of your gradient descent implementation.

Your next task is to complete the code in the file computeCost.m, which is a function that computes <em>J</em>(<em>θ</em>). As you are doing this, remember that the variables <em>X </em>and <em>y </em>are not scalar values, but matrices whose rows represent the examples from the training set.

Once you have completed the function, the next step in ex1.m will run computeCost once using <em>θ </em>initialized to zeros, and you will see the cost printed to the screen.

You should expect to see a cost of 32.07.

<em>You should now submit your solutions.</em>

<h3>2.2.4          Gradient descent</h3>

Next, you will implement gradient descent in the file gradientDescent.m. The loop structure has been written for you, and you only need to supply the updates to <em>θ </em>within each iteration.

As you program, make sure you understand what you are trying to optimize and what is being updated. Keep in mind that the cost <em>J</em>(<em>θ</em>) is parameterized by the vector <em>θ</em>, not <em>X </em>and <em>y</em>. That is, we minimize the value of <em>J</em>(<em>θ</em>) by changing the values of the vector <em>θ</em>, not by changing <em>X </em>or <em>y</em>. Refer to the equations in this handout and to the video lectures if you are uncertain.

A good way to verify that gradient descent is working correctly is to look at the value of <em>J</em>(<em>θ</em>) and check that it is decreasing with each step. The starter code for gradientDescent.m calls computeCost on every iteration and prints the cost. Assuming you have implemented gradient descent and computeCost correctly, your value of <em>J</em>(<em>θ</em>) should never increase, and should converge to a steady value by the end of the algorithm.

After you are finished, ex1.m will use your final parameters to plot the linear fit. The result should look something like Figure 2:

Your final values for <em>θ </em>will also be used to make predictions on profits in areas of 35,000 and 70,000 people. Note the way that the following lines in ex1.m uses matrix multiplication, rather than explicit summation or looping, to calculate the predictions. This is an example of code vectorization in Octave/MATLAB.

<em>You should now submit your solutions.</em>

predict1 = [1, 3.5] * theta; predict2 = [1, 7] * theta;

<h2>2.3           Debugging</h2>

Here are some things to keep in mind as you implement gradient descent: • Octave/MATLAB array indices start from one, not zero. If you’re storing <em>θ</em><sub>0 </sub>and <em>θ</em><sub>1 </sub>in a vector called theta, the values will be theta(1) and theta(2).

<ul>

 <li>If you are seeing many errors at runtime, inspect your matrix operations to make sure that you’re adding and multiplying matrices of compatible dimensions. Printing the dimensions of variables with the size command will help you debug.</li>

</ul>

Figure 2: Training data with linear regression fit

<ul>

 <li>By default, Octave/MATLAB interprets math operators to be matrix operators. This is a common source of size incompatibility errors. If you don’t want matrix multiplication, you need to add the “dot” notation to specify this to Octave/MATLAB. For example, A*B does a matrix multiply, while *B does an element-wise multiplication.</li>

</ul>

<h2>2.4           Visualizing <em>J</em>(<em>θ</em>)</h2>

To understand the cost function <em>J</em>(<em>θ</em>) better, you will now plot the cost over a 2-dimensional grid of <em>θ</em><sub>0 </sub>and <em>θ</em><sub>1 </sub>values. You will not need to code anything new for this part, but you should understand how the code you have written already is creating these images.

In the next step of ex1.m, there is code set up to calculate <em>J</em>(<em>θ</em>) over a grid of values using the computeCost function that you wrote.

<table width="527">

 <tbody>

  <tr>

   <td width="527">% initialize J vals to a matrix of 0’sJ vals = zeros(length(theta0vals), length(theta1vals));% Fill out J vals for i = 1:length(theta0vals) for j = 1:length(theta1vals) t = [theta0vals(i); theta1vals(j)];J vals(i,j) = computeCost(x, y, t); endend</td>

  </tr>

 </tbody>

</table>

After these lines are executed, you will have a 2-D array of <em>J</em>(<em>θ</em>) values. The script ex1.m will then use these values to produce surface and contour plots of <em>J</em>(<em>θ</em>) using the surf and contour commands. The plots should look something like Figure 3:

(a) Surface                                              (b) Contour, showing minimum

Figure 3: Cost function <em>J</em>(<em>θ</em>)

The purpose of these graphs is to show you that how <em>J</em>(<em>θ</em>) varies with changes in <em>θ</em><sub>0 </sub>and <em>θ</em><sub>1</sub>. The cost function <em>J</em>(<em>θ</em>) is bowl-shaped and has a global mininum. (This is easier to see in the contour plot than in the 3D surface plot). This minimum is the optimal point for <em>θ</em><sub>0 </sub>and <em>θ</em><sub>1</sub>, and each step of gradient descent moves closer to this point.

<h1>Optional Exercises</h1>

If you have successfully completed the material above, congratulations! You now understand linear regression and should able to start using it on your own datasets.

For the rest of this programming exercise, we have included the following optional exercises. These exercises will help you gain a deeper understanding of the material, and if you are able to do so, we encourage you to complete them as well.

<h1>3             Linear regression with multiple variables</h1>

In this part, you will implement linear regression with multiple variables to predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.

The file ex1data2.txt contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.

The ex1 multi.m script has been set up to help you step through this exercise.

<h2>3.1           Feature Normalization</h2>

The ex1 multi.m script will start by loading and displaying some values from this dataset. By looking at the values, note that house sizes are about 1000 times the number of bedrooms. When features differ by orders of magnitude, first performing feature scaling can make gradient descent converge much more quickly.

Your task here is to complete the code in featureNormalize.m to

<ul>

 <li>Subtract the mean value of each feature from the dataset.</li>

 <li>After subtracting the mean, additionally scale (divide) the feature values by their respective “standard deviations.”</li>

</ul>

The standard deviation is a way of measuring how much variation there is in the range of values of a particular feature (most data points will lie within ±2 standard deviations of the mean); this is an alternative to taking the range of values (max-min). In Octave/MATLAB, you can use the “std” function to compute the standard deviation. For example, inside featureNormalize.m, the quantity X(:,1) contains all the values of <em>x</em><sub>1 </sub>(house sizes) in the training set, so std(X(:,1)) computes the standard deviation of the house sizes. At the time that featureNormalize.m is called, the extra column of 1’s corresponding to <em>x</em><sub>0 </sub>= 1 has not yet been added to X (see ex1 multi.m for details).

You will do this for all the features and your code should work with datasets of all sizes (any number of features / examples). Note that each column of the matrix X corresponds to one feature.

<em>You should now submit your solutions.</em>

<strong>Implementation Note: </strong>When normalizing the features, it is important to store the values used for normalization – the <em>mean value </em>and the <em>standard deviation </em>used for the computations. After learning the parameters from the model, we often want to predict the prices of houses we have not seen before. Given a new <strong>x </strong>value (living room area and number of bedrooms), we must first normalize <strong>x </strong>using the mean and standard deviation that we had previously computed from the training set.

<h2>3.2           Gradient Descent</h2>

Previously, you implemented gradient descent on a univariate regression problem. The only difference now is that there is one more feature in the matrix X. The hypothesis function and the batch gradient descent update rule remain unchanged.

You should complete the code in computeCostMulti.m and gradientDescentMulti.m to implement the cost function and gradient descent for linear regression with multiple variables. If your code in the previous part (single variable) already supports multiple variables, you can use it here too.

Make sure your code supports any number of features and is well-vectorized. You can use ‘size(X, 2)’ to find out how many features are present in the dataset.

<em>You should now submit your solutions.</em>

<table width="516">

 <tbody>

  <tr>

   <td width="516"><strong>Implementation Note: </strong>In the multivariate case, the cost function can also be written in the following vectorized form:where             (1))<em>T </em>—                     <em>y</em>(1) — (<em>x</em> — (<em>x</em>(2))<em>T </em>— <em>X </em>=               ..                            <sub></sub><em>.</em>— (The vectorized version is efficient when you’re working with numerical computing tools like Octave/MATLAB. If you are an expert with matrix operations, you can prove to yourself that the two forms are equivalent.</td>

  </tr>

 </tbody>

</table>

<h3>3.2.1          Optional (ungraded) exercise: Selecting learning rates</h3>

In this part of the exercise, you will get to try out different learning rates for the dataset and find a learning rate that converges quickly. You can change the learning rate by modifying ex1 multi.m and changing the part of the code that sets the learning rate.

The next phase in ex1 multi.m will call your gradientDescent.m function and run gradient descent for about 50 iterations at the chosen learning rate. The function should also return the history of <em>J</em>(<em>θ</em>) values in a vector J. After the last iteration, the ex1 multi.m script plots the J values against the number of the iterations.

If you picked a learning rate within a good range, your plot look similar Figure 4. If your graph looks very different, especially if your value of <em>J</em>(<em>θ</em>) increases or even blows up, adjust your learning rate and try again. We recommend trying values of the learning rate <em>α </em>on a log-scale, at multiplicative steps of about 3 times the previous value (i.e., 0.3, 0.1, 0.03, 0.01 and so on). You may also want to adjust the number of iterations you are running if that will help you see the overall trend in the curve.

Figure 4: Convergence of gradient descent with an appropriate learning rate

<strong>Implementation Note: </strong>If your learning rate is too large, <em>J</em>(<em>θ</em>) can diverge and ‘blow up’, resulting in values which are too large for computer calculations. In these situations, Octave/MATLAB will tend to return NaNs. NaN stands for ‘not a number’ and is often caused by undefined operations that involve −∞ and +∞.

<table width="516">

 <tbody>

  <tr>

   <td width="516"><strong>Octave/MATLAB Tip: </strong>To compare how different learning learning rates affect convergence, it’s helpful to plot J for several learning rates on the same figure. In Octave/MATLAB, this can be done by performing gradient descent multiple times with a ‘hold on’ command between plots. Concretely, if you’ve tried three different values of alpha (you should probably try more values than this) and stored the costs in J1, J2 and J3, you can use the following commands to plot them on the same figure:plot(1:50, J1(1:50), ‘b’); hold on;plot(1:50, J2(1:50), ‘r’); plot(1:50, J3(1:50), ‘k’);The final arguments ‘b’, ‘r’, and ‘k’ specify different colors for the plots.</td>

  </tr>

 </tbody>

</table>

Notice the changes in the convergence curves as the learning rate changes. With a small learning rate, you should find that gradient descent takes a very long time to converge to the optimal value. Conversely, with a large learning rate, gradient descent might not converge or might even diverge!

Using the best learning rate that you found, run the ex1 multi.m script to run gradient descent until convergence to find the final values of <em>θ</em>. Next, use this value of <em>θ </em>to predict the price of a house with 1650 square feet and 3 bedrooms. You will use value later to check your implementation of the normal equations. Don’t forget to normalize your features when you make this prediction!

<em>You do not need to submit any solutions for these optional (ungraded) exercises.</em>

<h2>3.3           Normal Equations</h2>

In the lecture videos, you learned that the closed-form solution to linear regression is

Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no “loop until convergence” like in gradient descent.

Complete the code in normalEqn.m to use the formula above to calculate <em>θ</em>. Remember that while you don’t need to scale your features, we still need to add a column of 1’s to the X matrix to have an intercept term (<em>θ</em><sub>0</sub>). The code in ex1.m will add the column of 1’s to X for you.

<em>You should now submit your solutions.</em>

<em>Optional (ungraded) exercise: </em>Now, once you have found <em>θ </em>using this method, use it to make a price prediction for a 1650-square-foot house with 3 bedrooms. You should find that gives the same predicted price as the value you obtained using the model fit with gradient descent (in Section 3.2.1).


