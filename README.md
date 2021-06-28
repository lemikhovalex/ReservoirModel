Here is a simple Reservoir model, oil+water, inspired by [John Foster](https://johnfoster.pge.utexas.edu/PGE323M-ResEngineeringIII/course-mat/) 

You can install it as `pip install --upgrade petro_res_pack`


Short reminder for me on how to push to PyPi:

- `python setup.py sdist bdist_wheel`
- `twine check dist/*`
- `twine upload dist/*`
