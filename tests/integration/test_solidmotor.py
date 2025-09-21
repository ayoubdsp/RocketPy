from unittest.mock import patch


@patch("matplotlib.pyplot.show")
def test_solid_motor_info(mock_show, cesaroni_m1670):
    """Tests the SolidMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    cesaroni_m1670 : rocketpy.SolidMotor
        The SolidMotor object to be used in the tests.
    """
    assert cesaroni_m1670.info() is None
    assert cesaroni_m1670.all_info() is None
