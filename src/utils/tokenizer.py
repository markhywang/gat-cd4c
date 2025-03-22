from typing import Optional
import regex as re

"""
Tokenizer for natural language processing using the Byte Pair Encoding (BPE) algorithm.
Additionally, this tokenizer uses regex splitting to recognize specific language and punctuations as distinct elements.
This is to ensure the structure and semantic meaning of the text is more accurately captured.
"""
class BytePairTokenizer:
    
    def __init__(self) -> None:
        # (byte1, byte2) -> minted index
        self.merges = {}
        
        # tokenized index -> byte stream
        self.vocab = {}
        
        # What is the next index value to append to our vocabulary?
        self.index = 0
        
        # Regex split pattern used by GPT-4
        self.regex_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        
    def init_vocab(self) -> None:
        """Initialize vocabulary by using all standard UTF-8 mappings from 0-255"""
        self.vocab = {idx : bytes([idx]) for idx in range(0, 256)}
        self.index = 256
        
    def update_frequencies(self, byte_stream: list[int], byte_pair_frequencies: dict[tuple[int, int], int]) -> None:
        """Given a byte stream, update the frequencies of the number of occurrences of each byte pair (mutating)"""
        # If the byte stream is of size one or less, then no pairs exist
        if len(byte_stream) <= 1:
            return
        
        all_byte_pairs = list(zip(byte_stream, byte_stream[1:])) # Zip all consecutive pairs within the byte stream
        
        # Get all byte pair frequencies
        for byte_pair in all_byte_pairs:
            if byte_pair not in byte_pair_frequencies:
                byte_pair_frequencies[byte_pair] = 1
            else:
                byte_pair_frequencies[byte_pair] += 1
    
    def get_optimal_pair(self, byte_stream: list[int]) -> Optional[tuple[int, int]]:
        """Given a byte stream, get the pair of bytes with lowest corresponding merge index"""
        best_pair = None
        lowest_merge_idx = float('inf') # Store the global lowest merge index
        all_byte_pairs = list(zip(byte_stream, byte_stream[1:])) # Zip all consecutive pairs within the byte stream
        
        for byte_pair in all_byte_pairs:
            # If the byte pair is not in merges dictionairy, then skip
            if byte_pair not in self.merges:
                continue
            
            cur_merge_idx = self.merges[byte_pair]
            
            # If current merge index is lowest than global lowest index so far, then update accordingly
            if cur_merge_idx < lowest_merge_idx:
                lowest_merge_idx = cur_merge_idx
                best_pair = byte_pair
        
        # Return the optimal pair or signal that no more merge pairs exist (when best_pair is None)   
        return best_pair
    
    def merge(self, byte_stream: list[int], merge_pair: tuple[int, int]) -> list[int]:
        """Given a byte stream and byte pair, update the stream by merging each pair into a new index"""
        new_byte_stream = []
        n = len(byte_stream)
        i = 0
        
        while i < n:
            # If an optimal pair is found, then replace the pair with the new merged index
            if i < n - 1 and merge_pair == (byte_stream[i], byte_stream[i + 1]):
                new_byte_stream.append(self.merges[(byte_stream[i], byte_stream[i + 1])])
                i += 2
            # Otherwise, keep the current byte    
            else:
                new_byte_stream.append(byte_stream[i])
                i += 1
                
        return new_byte_stream
    
    def decode(self, ids: list[int]) -> str:
        """Converts list of tokenized IDs and decodes into string"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text: str) -> list[int]:
        """Converts string text into a list of tokenized IDs"""
        byte_stream = list(text.encode("utf-8"))
        
        while True:
            # Try to merge the pair of bytes that correspond with the lowest tokenized indices
            optimal_merge_pair = self.get_optimal_pair(byte_stream)
            
            # If no merge pair exists, then exist. Otherwise, merge accordingly
            if optimal_merge_pair is None:
                break
            else:
                # Update the current byte stream
                byte_stream = self.merge(byte_stream, optimal_merge_pair)
                
        return byte_stream
    
    def train(self, text: str, num_merges: int, init_utf=True, verbose=False) -> None:
        """Trains our byte-pair tokenizer on a given input text"""
        if init_utf and self.index == 0:
            self.init_vocab()
            
        # Split index with respect to regex pattern for individual training per element
        regex_split = re.findall(self.regex_pattern, text)
        
        # Get byte streams for the text
        byte_streams = [list(chunk.encode("utf-8")) for chunk in regex_split]
        n = len(byte_streams)
        
        # Assert that byte_streams is not empty
        assert n > 0
        
        # Peform byte-pair encoding by merging the most frequent byte-pairs and minting a new tokenization index
        for _ in range(0, num_merges):
            # If every individual byte stream has 1 byte or less, then stop BPE process
            if max(len(byte_stream) for byte_stream in byte_streams) <= 1:
                break
            
            byte_pair_frequencies = {} # (byte1, byte2) -> number of occurrences for this particular pair
            
            for byte_stream in byte_streams:
                # Perform in-place (mutating) update to byte pair frequencies
                self.update_frequencies(byte_stream, byte_pair_frequencies)
                
            # Get the most frequent byte pairs to prepare merging process
            most_frequent_pair = sorted(byte_pair_frequencies.items(), key=lambda x: x[1])[0][0]
            
            # If verbose=True, then output what is being merged
            if verbose:
                print(f"Merged {most_frequent_pair} -> {self.index}")
            
            # Update all attributes accordingly
            self.merges[most_frequent_pair] = self.index
            self.vocab[self.index] = self.vocab[most_frequent_pair[0]] + self.vocab[most_frequent_pair[1]]
            self.index += 1
            
            # Perform byte-pair merging and index minting process
            for i in range(0, n):
                byte_streams[i] = self.merge(byte_streams[i], most_frequent_pair)
