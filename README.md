# Master Thesis

To optimize the weights in an artificial neural network most methods rely
gradients, which are not always obtainable or desirable. Evolutionary algorithms
are instead based on Darwinian evolution where no derivative is needed. These
algorithms have a set of strategy parameters that can be dynamically updated during
the search to increase performance. Two ways of updating the parameters are the so
called “1/5th-rule”, which uses the offspring survival rate to self adapt, and random
mutation which uses inheritance and mutation to evolve the strategy parameters as
well.
We present an algorithm that combines the aspects of these two self adaptation
methods by changing strategy parameters differently for new offspring and older
survivors. We also introduce a pseudo-gradient by adding a memory of the previous
step taken in the search space and let the new mutation be shifted by this remembered
step. In this investigation these two methods failed to improve the performance
over the “1/5th-rule” but performed better than the random mutation. The new
algorithms showed promising results regarding combining the aspects of the
“1/5th-rule” and random mutation.
