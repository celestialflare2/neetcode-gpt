from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        tokenized_numbers = []
        for number in numbers:
            num_str = str(number)
            tokens = []
            i = 0
            while i < len(num_str):
                longest_match = None
                for j in range(i + 1, len(num_str) + 1):
                    substring = num_str[i:j]
                    if substring in vocab:
                        longest_match = substring
                if longest_match:
                    tokens.append(longest_match)
                    i += len(longest_match)
                else:
                    tokens.append(num_str[i])
                    i += 1
            tokenized_numbers.append(tokens)
        return tokenized_numbers

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        count = 0
        i = 0
        while i < len(text):
            longest_match = None
            for j in range(i + 1, len(text) + 1):
                substring = text[i:j]
                if substring in vocab:
                    longest_match = substring
            if longest_match:
                count += 1
                i += len(longest_match)
            else:
                count += 1
                i += 1
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        word_count = len(words)
        token_count = self.count_tokens(text, vocab)
        fertility = token_count / word_count if word_count > 0 else 0.0
        return round(fertility, 4)