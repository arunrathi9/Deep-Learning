# Deep-Learning
🥇 Father of Deep Learning - 🌟Geoffrey Hinton🌟

I'm using this repo to consolidate all my learning about Deep learning

## :rocket: Let's Get Started :fire:

## 📇 Content:

1. Foundation
2. Neural Networks
3. Artifical NN

## Topic 1: Foundation (covering useful Math concepts)

### Functions

For every function, I will cover three parts as:
 - Math formula
 - Diagram (x-y) plane
 - Code

**Math**
Example:

$f_1(x) = x^2;$ </br>
$f_2(x) = max(x,0);$ </br>
<img width="663" alt="square_ReLu" src="https://github.com/arunrathi9/Deep-Learning/assets/27626791/840c0999-9d31-47ef-9b37-deb9e4627a23">
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

- Derivative of a function at a point is the “rate of change” of the output of the function with respect to its input at that point.

Example:</br>

$\frac{\mathrm{d} f}{\mathrm{d} x}(x) = \displaystyle \lim_{a \to 0}\frac{f(x+a)-f(x-a)}{2*a}$


## Topic 1: Fundamentals of Deep Learning

### Intro

Major Breakthrough for DL - 2012 ImageNet Datasets - DL models won with a huge margin

Neural Networks: interconnected layers of artificial neurons that learn from data to make predictions or perform tasks. They excel at modeling complex patterns and have achieved significant breakthroughs in areas like image recognition, natural language processing, and speech synthesis.

![biological_neuron](https://github.com/arunrathi9/Deep-Learning/assets/27626791/c46a80a5-482a-4cce-9715-eedf4bc8ce70)
![artifical_neuron](https://github.com/arunrathi9/Deep-Learning/assets/27626791/8b8d6762-c2cf-4bc8-8d88-9ff48c86e4b2)



## Resources:

1. [Superdatascience website](https://www.superdatascience.com/courses/deep-learning-az)
2. [Writing Math formalae in Latex Style](https://editor.codecogs.com)
