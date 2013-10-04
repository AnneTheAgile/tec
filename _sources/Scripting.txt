Scripting System
================
The classes and API of the tec package's library were designed to create a generic interface separate from the implementation details of any particular model. While this design is useful, it doesn't completely address all of the requirements a user has. For example, many times a user is interested in seeing the output power density vs. voltage of a device who's other parameters remain fixed. Obviously, the user can write a script to iterate over a list of voltage values, set the voltage of a TEC object, then query the object for the output power density and record the result. This approach is a decent first step, but it has many problems:
  
* What should the structure of the output data be?
* How do we couple the script used to generate the data to the data itself for purposes of provenance?
* Recording only two arrays of data is throwing away a lot of useful metadata and computer time -- how do we capture that metadata and avoid re-calculating all that data?
* Is there an easy way of visualizing the data the script author intended to calculate?
  
The purpose of the tec package scripting system is to address all of those questions. The scripting system allows the user to write a script to generate data from the tec library so that the script itself and any TEC objects that are created are all put into a dict-like structure and saved to a file using the python pickle module. The scripting system also provides functionality so that the user can define a way to visualize the data that was generated. In this way no expensive data is lost during the execution of the script, the script itself is saved along with the data, and visualization of the data the script's author wanted is straightforward.
