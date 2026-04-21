"""
Indian License Plate Formatter
Validates and formats OCR output to match Indian license plate standards

Indian License Plate Format:
- State Code: 2 letters (e.g., MH, DL, KA, HR, PB, UP, TN, GJ, RJ)
- District Code: 2 digits (01-99)
- Series: 1-3 letters (A, AB, ABC)
- Number: 1-4 digits (1-9999)

Examples:
- MH 12 AB 1234
- DL 7C Q 1939
- KA 51 N 0099
"""
import re
from typing import Optional, Tuple

# Valid Indian state codes
INDIAN_STATE_CODES = [
    "AN",  # Andaman and Nicobar Islands
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CH",  # Chandigarh
    "CG",  # Chhattisgarh
    "DD",  # Daman and Diu
    "DL",  # Delhi
    "GA",  # Goa
    "GJ",  # Gujarat
    "HP",  # Himachal Pradesh
    "HR",  # Haryana
    "JH",  # Jharkhand
    "JK",  # Jammu and Kashmir
    "KA",  # Karnataka
    "KL",  # Kerala
    "LA",  # Ladakh
    "LD",  # Lakshadweep
    "MH",  # Maharashtra
    "ML",  # Meghalaya
    "MN",  # Manipur
    "MP",  # Madhya Pradesh
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha (also OR)
    "OR",  # Odisha (old code)
    "PB",  # Punjab
    "PY",  # Puducherry
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TN",  # Tamil Nadu
    "TS",  # Telangana
    "TR",  # Tripura
    "UK",  # Uttarakhand (also UA)
    "UA",  # Uttarakhand (old code)
    "UP",  # Uttar Pradesh
    "WB",  # West Bengal
]

# Common OCR misreadings and corrections
OCR_CORRECTIONS = {
    # Numbers misread as letters
    '0': 'O',  # Can be either
    'O': '0',  # Can be either
    '1': 'I',  # Can be either
    'I': '1',  # Can be either
    '5': 'S',
    'S': '5',
    '8': 'B',
    'B': '8',
    '6': 'G',
    'G': '6',
    '2': 'Z',
    'Z': '2',
    # Special characters to remove
    '@': 'A',
    '#': 'H',
    '$': 'S',
    '&': '8',
}


class IndianPlateFormatter:
    """Formats and validates Indian license plate numbers"""
    
    def __init__(self):
        self.state_codes = set(INDIAN_STATE_CODES)
        
        # Pattern for Indian license plate
        # Format: SS DD SSS NNNN (State-District-Series-Number)
        # The series can be 1-3 letters, number can be 1-4 digits
        self.plate_pattern = re.compile(
            r'^([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{1,3})\s*(\d{1,4})$'
        )
        
        # Alternative pattern for BH (Bharat) series plates
        self.bh_pattern = re.compile(
            r'^(\d{2})\s*BH\s*(\d{4})\s*([A-Z]{1,2})$'
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean OCR output text
        
        Args:
            text: Raw OCR output
            
        Returns:
            Cleaned text with only alphanumeric characters
        """
        # Convert to uppercase
        text = text.upper()
        
        # Collapse newlines and multi-line OCR output into a single line
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove common prefixes that might be detected
        prefixes_to_remove = ['IND', 'INDIA', 'AND', 'NON']
        for prefix in prefixes_to_remove:
            if text.startswith(prefix + ' '):
                text = text[len(prefix):].strip()
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Also remove these prefixes if they appear after the plate text (trailing junk)
        for suffix in ['NON', 'IND', 'INDIA']:
            if text.endswith(' ' + suffix):
                text = text[:-(len(suffix) + 1)].strip()
        
        # Replace common OCR errors
        for wrong, right in [('@', 'A'), ('#', 'H'), ('$', 'S'), ('/', ''), ('\\', ''), ('|', '1'), ('~', ''), ('`', '')]:
            text = text.replace(wrong, right)
        
        # Remove all non-alphanumeric characters except spaces
        text = re.sub(r'[^A-Z0-9\s]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_components(self, text: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Extract license plate components from text
        
        Args:
            text: Cleaned text
            
        Returns:
            Tuple of (state_code, district_code, series, number) or None
        """
        text = self.clean_text(text)
        
        # Remove all spaces for pattern matching
        compact = text.replace(' ', '')
        
        # Heuristic: if raw text is already valid, we should still try corrections
        # because OCR might read digits as letters (e.g. G -> 6) which fits regex but is wrong.
        # So we calculate corrected version first.
        
        corrected = self._apply_ocr_corrections(compact)
        
        # Check corrected version
        match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', corrected)
        if match:
            state, district, series, number = match.groups()
            
            # Try alt corrections (0->Q instead of 0->O) to see if it produces
            # a better result — both must be valid, pick the one with valid state
            alt_corrected = self._apply_ocr_corrections_alt(compact)
            alt_match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', alt_corrected)
            
            if alt_match:
                alt_state, alt_dist, alt_series, alt_num = alt_match.groups()
                # Prefer alt only if it changes the series (Q vs O) and
                # the alt state is valid (or both states are equally valid)
                if alt_series != series:
                    primary_state_valid = state in self.state_codes
                    alt_state_valid = alt_state in self.state_codes
                    if alt_state_valid and not primary_state_valid:
                        return alt_match.groups()
            
            return match.groups()
             
        # If corrected verification failed, check if raw was valid
        match = re.search(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', compact)
        if match:
             return match.groups()

        # Try OCR-corrected version with search (not anchored) - fallback
        match = re.search(r'([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})', corrected)
        if match:
             return match.groups()
        
        return None
    
    def _apply_ocr_corrections_alt(self, text: str) -> str:
        """
        Alternative OCR corrections are no longer used since we rely on the
        strict user-defined positional logic and the advanced candidate ranking
        in video_pipeline.py. We simply return the primary strict corrections.
        """
        return self._apply_ocr_corrections(text)
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """
        Apply strict positional OCR corrections based on user logic:
        - First 2 characters must be letters
        - 3rd character must be a number
        - Last 4 characters must be numbers
        - 5th character from last must be a letter
        - Middle characters left ambiguous
        """
        if len(text) < 7:
            return text
            
        result = list(text)
        
        digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
        letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'T': '7', 'D': '0', 'Q': '0'}
        
        # 1. First 2 characters must be letters
        for i in range(2):
            if result[i] in digit_to_letter:
                result[i] = digit_to_letter[result[i]]
                
        # 2a. 3rd character must be a number
        if result[2] in letter_to_digit:
            result[2] = letter_to_digit[result[2]]
            
        # 2b. 4th character heuristic: if it strongly resembles a digit, assume it's
        # the second half of a 2-digit district code. (Fixes HR 2S -> HR 25).
        # Otherwise, if it resembles a letter, it must be the start of the series.
        if len(result) > 3:
            if result[3] in letter_to_digit:
                result[3] = letter_to_digit[result[3]]
            elif result[3] in digit_to_letter:
                result[3] = digit_to_letter[result[3]]
                
        # 2c. Middle characters (index 4 up to the padding) MUST be letters (Series region)
        for i in range(4, len(result) - 5):
            if result[i] in digit_to_letter:
                result[i] = digit_to_letter[result[i]]
            
        # 3. Last 4 characters must be numbers
        for i in range(len(result) - 4, len(result)):
            if result[i] in letter_to_digit:
                result[i] = letter_to_digit[result[i]]
                
        # 4. 5th character from last must be a letter
        if len(result) >= 5:
            if result[-5] in digit_to_letter:
                result[-5] = digit_to_letter[result[-5]]
                
        return ''.join(result)
    
    def format_plate(self, text: str) -> str:
        """
        Format text as a standard Indian license plate
        
        Args:
            text: OCR output text
            
        Returns:
            Formatted plate number (e.g., "MH 12 AB 1234")
        """
        components = self.extract_components(text)
        
        if components:
            state, district, series, number = components
            
            # Validate state code
            if state not in self.state_codes and state != 'BH':
                # Try to find closest match
                state = self._find_closest_state(state)
            
            # Keep district and number as-is (no zero padding)
            
            return f"{state} {district} {series} {number}"
        
        # If pattern matching failed, just clean and return
        return self.clean_text(text)
    
    def _find_closest_state(self, code: str) -> str:
        """Find the closest matching state code"""
        if len(code) != 2:
            return code
        
        # Check for common OCR errors in state codes
        corrections = {
            'NH': 'MH',  # Maharashtra
            'OK': 'DL',  # Delhi
            'KK': 'KA',  # Karnataka
            'RI': 'RJ',  # Rajasthan
            'OL': 'DL',  # Delhi
        }
        
        if code in corrections:
            return corrections[code]
        
        return code
    
    def validate_plate(self, text: str) -> Tuple[bool, str]:
        """
        Validate if text is a valid Indian license plate
        
        Args:
            text: Plate text to validate
            
        Returns:
            Tuple of (is_valid, formatted_plate)
        """
        components = self.extract_components(text)
        
        if not components:
            return False, self.clean_text(text)
        
        state, district, series, number = components
        
        # Validate state code
        is_valid_state = state in self.state_codes or state == 'BH'
        
        # Validate district (1-99)
        try:
            dist_num = int(district)
            is_valid_district = 1 <= dist_num <= 99
        except ValueError:
            is_valid_district = False
        
        # Validate number (1-9999)
        try:
            num = int(number)
            is_valid_number = 1 <= num <= 9999
        except ValueError:
            is_valid_number = False
        
        is_valid = is_valid_state and is_valid_district and is_valid_number
        formatted = self.format_plate(text)
        
        return is_valid, formatted


def format_indian_plate(text: str) -> str:
    """
    Convenience function to format an Indian license plate
    
    Args:
        text: Raw OCR output
        
    Returns:
        Formatted Indian license plate
    """
    formatter = IndianPlateFormatter()
    return formatter.format_plate(text)


def validate_indian_plate(text: str) -> Tuple[bool, str]:
    """
    Convenience function to validate an Indian license plate
    
    Args:
        text: Raw OCR output
        
    Returns:
        Tuple of (is_valid, formatted_plate)
    """
    formatter = IndianPlateFormatter()
    return formatter.validate_plate(text)


# Test the formatter
if __name__ == "__main__":
    formatter = IndianPlateFormatter()
    
    test_cases = [
        "IND MH 12 AB 3456",
        "AND DL 7CQ 1939",
        "MH43CC1745/",
        "KA 64 N 0099",
        "MH20 DV2363",
        "PB46 DZ687",
        "MHG7AG4423",
        "WP53@VGOOD",
        "#R123C0547 1",
    ]
    
    print("Indian License Plate Formatter Test")
    print("=" * 50)
    
    for test in test_cases:
        is_valid, formatted = formatter.validate_plate(test)
        status = "✓" if is_valid else "✗"
        print(f"{status} '{test}' -> '{formatted}'")
