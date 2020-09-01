from steps import Stepper, UniformSteps


def test_basic_init():

    assert isinstance(Stepper(), Stepper)
    assert isinstance(UniformSteps(), Stepper)
