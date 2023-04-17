"""
"""

from configparser import ConfigParser, ExtendedInterpolation


class Configuration(ConfigParser):
    """
    This class provides an interface to read and write configuration files in
    INI format. It inherits from ConfigParser, which is a built-in Python
    class for working with configuration files.

    Args:
        path_to_ini_file: The path to the INI file.

    Attributes:
        path_to_ini (str): The path to the INI file.
    """

    def __init__(self, path_to_ini_file: str) -> None:
        """
        Initializes a new instance of the Configuration class.

        Args:
            path_to_ini_file (str): The path to the INI file.
        """
        self.path_to_ini = path_to_ini_file

        # Initialize the base class with interpolation and read the INI file.
        super().__init__(interpolation=ExtendedInterpolation())
        self.read(path_to_ini_file)

    def get_list(self, section: str, option: str) -> list:
        """
        Reads a list from a section.

        Args:
            section (str): The name of the section.
            option (str): The name of the option.

        Returns:
            list: The list of values.
        """
        # Get the value and map it to a list.
        value = self.get(section, option)
        return list(filter(None, (x.strip() for x in value.splitlines())))

    def update_values(self, section: str, option: str, value: str) -> bool:
        """
        Updates the values of an option in a section.

        Args:
            section (str): The name of the section.
            option (str): The name of the option.
            value (str): The new value of the option.

        Returns:
            bool: True if the update was successful, otherwise False.
        """
        self.set(section, option, value)

        # Write the changes to the INI file.
        with open(self.path_to_ini, "w") as fh:
            self.write(fh)

        return True
