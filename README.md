# PaperWings

An experimental Vector Symbolic Architecture (VSA) associative memory implementation in Python that features biologically-inspired forgetting mechanisms and advanced unbinding operations.

## Table of Contents
- [Overview](#overview)
- [What are VSAs and Hyperdimensional Computing?](#what-are-vsas-and-hyperdimensional-computing)
- [Features](#features)
- [Usage Example](#usage-example)
- [Memory and Forgetting](#memory-and-forgetting)
- [Name Origin](#name-origin)
- [License](#license)
- [Disclaimer](#disclaimer)

## Overview

PaperWings is an implementation of Hyperdimensional Computing (HDC) and Vector Symbolic Architectures (VSA), which are computational frameworks inspired by the brain's ability to process information using high-dimensional representations. The project implements two key biologically-inspired features:
- Memory decay that simulates how human memories naturally fade over time
- Unbinding operations that can recover individual concepts from compound representations, similar to how humans can extract specific details from complex memories

This implementation provides tools for creating and manipulating high-dimensional vectors to build associative memories and semantic representations, with a focus on cognitive-like computing capabilities.

## What are VSAs and Hyperdimensional Computing?

Vector Symbolic Architectures are mathematical frameworks that use high-dimensional vectors (typically 1000+ dimensions) to represent and manipulate symbolic information. These architectures are based on three key operations:
- **Binding**: Combining two vectors to create a new vector that represents their association
- **Bundling**: Adding vectors to create a composite representation
- **Similarity**: Measuring how similar two vectors are to each other

Hyperdimensional Computing leverages these operations to perform cognitive computing tasks like analogical reasoning, semantic composition, and associative memory.

## Features

- Binary and Binary Sparse vector representations
- Vector space operations (binding, bundling, similarity search)
- Triple unbinding for knowledge graph-like operations
- Associative memory with decay functionality
- Parallel processing support for vector operations

## Usage Example

Here's a simple example of creating semantic relationships using PaperWings:

```python
from paperwings.vector.vector_space import VectorSpace

# Create a vector space with binary vectors of 1000 dimensions
space = VectorSpace(rep="binary")

# Create vectors for concepts
country = space.add_vector("COUNTRY")
currency = space.add_vector("CURRENCY")
usa = space.add_vector("USA")
dollar = space.add_vector("DOLLAR")
mexico = space.add_vector("MEXICO")
pesos = space.add_vector("PESOS")

# Create composite representations
usa_currency = country * usa + currency * dollar  # USA uses Dollar
mexico_currency = country * mexico + currency * pesos  # Mexico uses Pesos

# Store in associative memory
space.insert_vector(usa_currency, "USA Currency")
space.insert_vector(mexico_currency, "Mexican Currency")

# Query the memory
result = space.find_vector(usa_currency)
assert result[0] == "USA Currency"
```

### Knowledge Graph Example

PaperWings can also represent and query more complex semantic relationships:

```python
from paperwings.vector.vector_space import VectorSpace
from paperwings.unbinder.triple_unbinder import TripleUnbinder

# Create a vector space for knowledge representation
space = VectorSpace(size=1000, rep="binary_sparse")

# Create vectors for a simple ontology
socrates = space.add_vector("Socrates")
is_a = space.add_vector("is_a")
man = space.add_vector("man")

# Bind them into a triple representation
bound_triple = socrates * is_a * man

# Create an unbinder to recover the original concepts
unbinder = TripleUnbinder(space, early_stop=True, top_k=10)
recovered_factors = unbinder.unbind(bound_triple)
```

## Memory and Forgetting

PaperWings implements a biologically-inspired decay mechanism. The associative memory can "forget" information over time using the `decay()` method, simulating how biological memories fade:

```python
# Information will be forgotten after multiple decay operations
space.decay()  # Call multiple times to simulate passage of time
```

## Name Origin

Named after the lyrics of Juanita/Kiteless by Underworld:

https://www.youtube.com/watch?v=-UCI-3xLewc

```
Your rails, your thin
Paper wings, paper wings
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is an experimental prototype implementation of VSA concepts. It's intended for research and experimentation rather than production use.
