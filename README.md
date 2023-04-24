# Bayesian Optimization Library (BOlite)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

BOLite is a small, illustrative library for Bayesian Optimization, designed for educational purposes. It provides an easy-to-understand implementation of the Bayesian Optimization algorithm to help users gain a deeper understanding of the concepts and techniques involved.

## Table of Contents

-[Installation](#installation)
-[Usage](#usage)
-[Examples](#examples)
-[Contributing](#contributing)
-[License](#license)

## Installation

To install the BOlite package, run the following command in the root folder:

`pip install -e .`

## Usage

To use BOlite, follow these simple steps:

1. Import the necessary classes:

`python from bolite.BayesOpt import BayesianOptimizer, GaussianProcess`

2. Define your objective function:

```
def objective_function(x):
    return -x**2 + 4*x - 4
```

3. Set up the Bayesian Optimization object:

`optimizer = BayesianOptimizer(objective_function, bounds=[(0, 4)])`

4. Run the optimization:

`optimizer.optimize(iterations=10)`

## Examples

For more examples and detailed explanations, visit the [examples](./examples) directory.

## Contributing

If you want to contribute to this project, please read the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines. We appreciate any help, from fixing bugs to improving documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
