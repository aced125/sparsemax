=========
sparsemax
=========


.. image:: https://img.shields.io/pypi/v/sparsemax.svg
        :target: https://pypi.python.org/pypi/sparsemax

.. image:: https://img.shields.io/travis/aced125/sparsemax.svg
        :target: https://travis-ci.com/aced125/sparsemax

.. image:: https://readthedocs.org/projects/sparsemax/badge/?version=latest
        :target: https://sparsemax.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/aced125/sparsemax/shield.svg
     :target: https://pyup.io/repos/github/aced125/sparsemax/
     :alt: Updates



A PyTorch implementation of SparseMax (https://arxiv.org/pdf/1602.02068.pdf) with gradients checked and tested


* Free software: MIT license
* Documentation: https://sparsemax.readthedocs.io.


Installation
------------

.. code-block:: bash

   pip install -U sparsemax


Usage
-----

Use as if it was :code:`nn.Softmax()`! Nice and simple.

.. code-block:: python

    from sparsemax import Sparsemax
    import torch
    import torch.nn as nn

    sparsemax = Sparsemax(dim=-1)
    softmax = torch.nn.Softmax(dim=-1)

    logits = torch.randn(2, 3, 5)
    logits.requires_grad = True
    print("\nLogits")
    print(logits)

    softmax_probs = softmax(logits)
    print("\nSoftmax probabilities")
    print(softmax_probs)

    sparsemax_probs = sparsemax(logits)
    print("\nSparsemax probabilities")
    print(sparsemax_probs)


Check that gradients are computed correctly
-------------------------------------------

.. code-block:: python

    from torch.autograd import gradcheck
    from sparsemax import Sparsemax

    input = (torch.randn(6, 3, 20,dtype=torch.double,requires_grad=True))
    test = gradcheck(sparsemax, input, eps=1e-6, atol=1e-4)
    print(test)



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
