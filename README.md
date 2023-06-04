# Deep-Learning
🥇 Father of Deep Learning - 🌟Geoffrey Hinton🌟

I'm using this repo to consolidate all my learning about Deep learning

## :rocket: Let's Get Started :fire:

## 📇 Content:

1. Foundation
2. Neural Networks
3. Artifical NN

## Topic 1: Foundation (covering useful Math concepts)

For every function & Derivatives, I will cover below parts as:
 - Math formula
 - Diagram (x-y) plane
 - Mini-factories diagram
 - Code

### Functions

**Math**
Example:

$f_1(x) = x^2;$ </br>
$f_2(x) = max(x,0);$ </br>
<img width="663" alt="square_ReLu" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/840c0999-9d31-47ef-9b37-deb9e4627a23">
<img width="659" alt="square_ReLu_factory" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/42a326d6-bb60-4171-9269-7be3e358c2f7">

```
def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input ndarray.
    '''
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    '''
    Apply "Leaky ReLU" function to each element in ndarray.
    '''
    return np.maximum(0.2 * x, x)
```

### Derivatives

- Derivative of a function at a point is the **“rate of change”** of the output of the function with respect to its input at that point.

Example:</br>

$\frac{\mathrm{d} f}{\mathrm{d} x}(x) = \displaystyle \lim_{a \to 0}\frac{f(x+a)-f(x-a)}{2*a}$

This limit i.e "a" can be approximated numerically by setting a very small value for a, such as 0.001, so we can compute the derivative.

<img width="655" alt="derivative_slope" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/87148696-0165-4147-a807-3de3c5cde9b5">
<img width="554" alt="derivative_factory" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/e26c41a8-4060-4cea-9640-e9ad1b006466">

```
from typing import Callable

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the
    "input_" array.
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)
```

### Nested Functions
If we have two functions that by mathematical convention we call f1 and f2, the output of one of the functions becomes the input to the next one, so that we can “string them together.”

- $f_2(f_1(x))$

<img width="659" alt="nested_function" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/5a05a52f-218a-48e5-805d-a5b6ffbc03e7">

```
from typing import List

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]

def chain_length_2(chain: Chain,
                   a: ndarray) -> ndarray:
    '''
    Evaluates two functions in a row, in a "Chain".
    '''
    assert len(chain) == 2, \
    "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))
```

<img width="575" alt="another_nested_function" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/a4fba39d-7535-4311-88fb-bb7eefd14869">

### The Chain Rule
 - a mathematical theorem that lets us compute derivatives of composite functions
<img width="286" alt="chain_rule" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/1dd35f9c-2f93-4d62-900f-5f23c46a312d">
<img width="675" alt="chain_rule_factory" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/b1cb931d-6242-4896-87f9-aae7e04ed712">

```
def sigmoid(x: ndarray) -> ndarray:
    '''
    Apply the sigmoid function to each element in the input ndarray.
    '''
    return 1 / (1 + np.exp(-x))
    
def chain_deriv_2(chain: Chain,
                  input_range: ndarray) -> ndarray:
    '''
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    '''

    assert len(chain) == 2, \
    "This function requires 'Chain' objects of length 2"

    assert input_range.ndim == 1, \
    "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx
    f1_of_x = f1(input_range)

    # df1/du
    df1dx = deriv(f1, input_range)

    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))

    # Multiplying these quantities together at each point
    return df1dx * df2du
    
#plots the results and shows that the chain rule works
PLOT_RANGE = np.arange(-3, 3, 0.01)

chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]

plot_chain(chain_1, PLOT_RANGE)
plot_chain_deriv(chain_1, PLOT_RANGE)

plot_chain(chain_2, PLOT_RANGE)
plot_chain_deriv(chain_2, PLOT_RANGE)
```

The chain rule seems to be working. When the functions are upward-sloping, the derivative is positive; when they are flat, the derivative is zero; and when they are downward-sloping, the derivative is negative.
</br>
<img width="672" alt="graph_nested_function" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/d0d31cec-3599-44ac-a62c-3fda5224cef2">

</br>
Example 2:<br>

<img width="438" alt="nested_3_function" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/7acfddc0-0e56-4622-bcca-4f9bf7f4b2a7">
<img width="672" alt="nested_3_function_factory" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/d8bdcbf5-c0f7-4562-8459-86ea304684d3">
</br>

```
def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:
    
    #Uses the chain rule to compute the derivative of three nested functions:
    #(f3(f2(f1)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    

    assert len(chain) == 3, \
    "This function requires 'Chain' objects to have length 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_of_x = f1(input_range)

    # f2(f1(x))
    f2_of_x = f2(f1_of_x)

    # df3du
    df3du = deriv(f3, f2_of_x)

    # df2du
    df2du = deriv(f2, f1_of_x)

    # df1dx
    df1dx = deriv(f1, input_range)

    # Multiplying these quantities together at each point
    return df1dx * df2du * df3du
    
PLOT_RANGE = np.range(-3, 3, 0.01)
plot_chain([leaky_relu, sigmoid, square], PLOT_RANGE)
plot_chain_deriv([leaky_relu, sigmoid, square], PLOT_RANGE)
```
 
Something interesting took place here—to compute the chain rule for this nested function, we made two “passes” over it:

- First, we went “forward” through it, computing the quantities f1_of_x and f2_of_x along the way. We can call this (and think of it as) “the forward pass.”
- Then, we “went backward” through the function, using the quantities that we computed on the forward pass to compute the quantities that make up the derivative.
Finally, we multiplied three of these quantities together to get our derivative.

Now, let’s show that this works, using the three simple functions we’ve defined so far: sigmoid, square, and leaky_relu.
<img width="420" alt="nested_3_function_diagram" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/289d12c7-c0c3-4e38-b456-377dc189bcd4">

### Functions with Multiple Inputs:
<img width="638" alt="multiple_input_factory" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/fbcef0e1-7300-480f-9402-1481cda41d95">
<img width="632" alt="multiple_input_derivative" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/3eda42ac-85ac-4cff-b6fa-450c92a606e1">


### Functions with Multiple Vector Inputs:
- In deep learning, we deal with functions whose inputs are vectors or matrices. 

Example of Housing price:<br>
- A typical way to represent a single data point, or “observation,” in a neural network is as a row with n features, where each feature is simply a number x1, x2, and so on, up to xn:
   - $x = [x_1, x_2, x_3, ... , x_n]$
   - x1, x2, and so on are numerical features of a house, such as its square footage or its proximity to schools.
- Creating New Features from Existing Features:
  - single most common operation in neural networks is to form a “weighted sum” of these features, where the weighted sum could emphasize certain features and de-emphasize others and thus be thought of as a new feature that itself is just a combination of old features.
  - weight vector: $$w = \begin{bmatrix}w_1\\w_2\\.\\.\\w_n\end{bmatrix}$$
  - Dot product of W and variable (X) vectors: $N = \upsilon (X, W)=X\times W =x_1\times w_1+x_2\times w_2+x_3\times w_3+...$

<img width="698" alt="matrix_multiplication" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/d48aa8a2-cf3c-4487-9e28-52448a75fe48">

```
def matmul_forward(X: ndarray,
                   W: ndarray) -> ndarray:
    
    #Computes the forward pass of a matrix multiplication.

    assert X.shape[1] == W.shape[0], \
    #```
    #For matrix multiplication, the number of columns in the first array should
    #match the number of rows in the second; instead the number of columns in the
    #first array is {0} and the number of rows in the second array is {1}.
    #'''.format(X.shape[1], W.shape[0])

    # matrix multiplication
    N = np.dot(X, W)

    return N
```

### Derivatives of Functions with Multiple Vector Inputs:


## Topic 1: Fundamentals of Deep Learning

### Intro

Major Breakthrough for DL - 2012 ImageNet Datasets - DL models won with a huge margin

Neural Networks: interconnected layers of artificial neurons that learn from data to make predictions or perform tasks. They excel at modeling complex patterns and have achieved significant breakthroughs in areas like image recognition, natural language processing, and speech synthesis.

![biological_neuron](https://github.com/arunrathi9/Deep-Learning/assets/27626791/c46a80a5-482a-4cce-9715-eedf4bc8ce70)
![artifical_neuron](https://github.com/arunrathi9/Deep-Learning/assets/27626791/8b8d6762-c2cf-4bc8-8d88-9ff48c86e4b2)



## Resources:

1. [Superdatascience website](https://www.superdatascience.com/courses/deep-learning-az)
2. [Writing Math formalae in Latex Style](https://editor.codecogs.com)
