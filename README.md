# Option Pricing Using Physics Informed Neural Networks

- [PINN module](./PINN/)
  - [VanillaOptionPINN](./PINN/VanillaOptions.py): PINN for non-dividend-paying European/American options
  - [utilities](./PINN/utilities.py): Crank-Nicolson method for European/American option pricing PDE, QuantLib pricing wraper (CRR, LSMC)
- [PINN training notebook](./PINN_training.ipynb)
- [PINN train loop](./train_loop.py): parallised PINN train loop
