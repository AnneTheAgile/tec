Quickstart
----------
TEC models are built by subclassing the TECBase class and adding functionality to calculate the motive. As a quick example, say the user wanted to simulate the device described in <REF - H&G>::

	>>> # Pull in Langmuir's model.
	>>> from tec.models import Langmuir

	>>> # Create a dict containing the emitter parameters (assume emissivity = 0.5).
	>>> em = {"temp":1,
	      "barrier":1,
	      "voltage":1,
	      "position":0,
	      "richardson":10,
	      "emissivity":0.5}

	>>> # Create the collector parameters.
	>>> em = {"temp":1,
	>>>       "barrier":1,
	>>>       "voltage":1,
	>>>       "position":0,
	>>>       "richardson":10,
	>>>       "emissivity":0.5}

	>>> # Put everything together to instantiate the TEC.
	>>> tec_params = {"Emitter":em, "Collector":co}
	>>> T = Langmuir(tec_params)

	>>> # What's the maximum output power density of this thing?
	>>> T.calc_output_power_density(with_output="max")

	>>> # What if we want to know the output power density if the device
	>>> # is operating at maximum efficiency? First, calculate the maximum
	>>> # efficiency and set the TEC to the corresponding voltage.
	>>> T.calc_total_efficiency(with_output="max",set_voltage=True)

	>>> # And look, the output voltage was set such that the TEC is at max efficiency.
	>>> T.calc_output_voltage()

	>>> # Now let's calculate the value of maximum output power.
	>>> T.calc_output_power_density()

	>>> # What does the motive diagram at this voltage look like?
	>>> T.plot_motive()

	>>> # Looks like a lot of negative space charge. What if we decrease the 
	>>> # interelectrode spacing?
	>>> T["Collector"]["position"] = 5
	>>> T.calc_total_efficiency(with_output="max",set_voltage=True)
	>>> T.calc_output_power_density()
	>>> T.plot_motive()
