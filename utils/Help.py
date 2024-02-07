class ColoredOutput:
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'PURPLE': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m'
    }

    @staticmethod
    def print_message(message, color='RESET', bold=False):
        color_code = ColoredOutput.COLORS.get(color, ColoredOutput.COLORS['RESET'])
        bold_code = ColoredOutput.COLORS['BOLD'] if bold else ''
        reset_code = ColoredOutput.COLORS['RESET']
        print(f'{bold_code}{color_code}{message}{reset_code}')

    @staticmethod
    def log_message(message, color='RESET', bold=False):
        color_code = ColoredOutput.COLORS.get(color, ColoredOutput.COLORS['RESET'])
        bold_code = ColoredOutput.COLORS['BOLD'] if bold else ''
        reset_code = ColoredOutput.COLORS['RESET']
        log_entry = f'{bold_code}{color_code}{message}{reset_code}'
        print(log_entry)

# ColoredOutput.print_message("This is a colored message", color='GREEN', bold=True)
# ColoredOutput.log_message("This is a log entry", color='BLUE')
