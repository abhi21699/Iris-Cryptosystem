import numpy as np
import galois
import try_FAR
import random
from sympy import GF
from itertools import combinations
import os
GF = GF(2**16 + 1)


def bch_encode(binary_list, n, k):
    """
    Encode a binary list using BCH encoding.
    Parameters:
        binary_list (list[int]): Input binary list of size k.
        n (int): Length of the BCH codeword.
        k (int): Length of the input binary message.
    Returns:
        list[int]: Encoded binary list of size n.
    """
    # Define the Galois field (GF)
    GF = galois.GF(2**16)
    
    # Define the BCH code parameters
    bch = galois.BCH(n, k)
    
    # Convert binary list to GF elements
    message = GF(binary_list)
    
    # Encode the binary list
    codeword = bch.encode(message)
    return codeword.tolist()

def bch_decode(codeword, n, k):
    """
    Decode a BCH-encoded binary list.
    Parameters:
        codeword (list[int]): Encoded binary list of size n.
        n (int): Length of the BCH codeword.
        k (int): Length of the original binary message.
    Returns:
        list[int]: Decoded binary list of size k.
    """
    # Define the Galois field (GF)
    GF = galois.GF(2**16)
    
    # Define the BCH code parameters
    bch = galois.BCH(n, k)
    
    # Convert codeword to GF elements
    codeword_gf = GF(codeword)
    
    # Decode the codeword
    decoded_message = bch.decode(codeword_gf)
    return decoded_message.tolist()
def xor_vectors(v1, v2):
    return [a ^ b for a, b in zip(v1, v2)]
def combineVectorbits(v):
    l = len(v)
    o = []
    for i in range(l):
        x = v[i]
        for j in range(3):
            o.append(int(x[j]))
    return o
# Partition Feature Vector
def partition_feature_vector(feature_vector, partition_size):
    return [feature_vector[i:i + partition_size] for i in range(0, len(feature_vector), partition_size)]

def add_chaff_points(num_chaff, genuine_points, field_size=2**16 +1):
    """
    Add random chaff points to the vault.
    """
    genuine_x = {x for x, _ in genuine_points}  # Genuine x-coordinates
    chaff_points = []
    while len(chaff_points) < num_chaff:
        x_c = random.randint(1, field_size - 1)
        if x_c not in genuine_x:  # Ensure no collision with genuine points
            y_c = random.randint(1, field_size - 1)
            chaff_points.append((x_c, y_c))
    return chaff_points
# # Example Usage
# if __name__ == "__main__":
#     # Input binary list of size k
#     binary_list = [1, 0, 1, 1, 0, 1, 0, 1,1, 0, 1, 1, 0, 1, 0, 1]
#     k = len(binary_list)  # Length of the input binary list
#     n = 31 # Length of the BCH codeword

#     # Encode
#     encoded = bch_encode(binary_list, n, k)
#     print("Encoded Binary List (n bits):", encoded)

#     # Decode
#     decoded = bch_decode(encoded, n, k)
#     print("Decoded Binary List (k bits):", decoded)

# Create Fuzzy Vault
def create_fuzzy_vault(feature_vector, degree=7, num_chaff=0,part_size=16):

    """
    Create a fuzzy vault for the input feature vector.
    Parameters:
        feature_vector (list[int]): Input binary feature vector (6144 bits).
        degree (int): Degree of the polynomial.
        num_chaff (int): Number of chaff points.
    Returns:
        tuple: Vault, polynomial coefficients, and transformed iriscode.
    """
    # Partition feature vector into 192 components of 32 bits each
    partitions = partition_feature_vector(feature_vector, part_size)
    print("orig vector: ", partitions)
    # Generate random K1 of size 16 *384 bits
    k1 = [random.randint(0, 1) for _ in range(6144)] # 16 * 192

    partitions_k1 = partition_feature_vector(k1,16)
    encoded_k1= partitions_k1
    print("original key k1: ", partitions_k1)
# use this bch encoding 
    # for i in range(len(partitions_k1)):
    #     encode_k1 = bch_encode(partitions_k1[i])
    #     # print("encoding: ", encode_k1 )
    #     encoded_k1.append(encode_k1)
    
    # XOR each partition with BCH codeword to form I_T*
    transformed_iriscode = [
        xor_vectors(part, list(map(int, encoded_part))) for part, encoded_part in zip(partitions, encoded_k1)
    ]
    vault, polynomial_coeffs = secure_k1_with_polynomial(k1,degree, num_chaff)
    return vault, polynomial_coeffs, transformed_iriscode
    

def secure_k1_with_polynomial(k1, degree, num_chaff=0):

    
    k2=  [random.randint(0, 1) for _ in range(128)]
    partitions_k1 = partition_feature_vector(k1,16)      # CHANGE PARTITION SIZE HERE  Also
    print("k1 partitions again: ", partitions_k1)
    print("---------------------------------------------------------------------------------------------")

    genuine_points, polynomial_coeffs = evaluate_polynomial(k2,partitions_k1,degree)
    chaff_points = add_chaff_points(num_chaff, genuine_points)
    # Create vault
    vault = genuine_points + chaff_points
    # random.shuffle(vault)

    return vault, polynomial_coeffs


def evaluate_polynomial(k2, k1_partitions, degree):
    """
    Evaluate a polynomial over GF(2^16) using partitions of k2 as coefficients and k1 partitions as points.
    Parameters:
        k2 (list[int]): Binary list representing coefficients of the polynomial.
        k1_partitions (list[list[int]]): Binary partitions of k1, each representing a point to evaluate the polynomial.
        degree (int): Degree of the polynomial.
    Returns:
        list[tuple]: A list of tuples (x, y), where y = f(x), with calculations in GF(2^16).
    """
    # Define the finite field GF(2^16)
    
    # k2 = 16 * 8 = 128 bits
    # Partition k2 into coefficients for the polynomial
    if len(k2) < (degree + 1) * 16:
        raise ValueError(f"k2 must have at least {(degree + 1) * 16} bits for degree {degree}")
    print("k2: ", k2)
    k2_coefficients = [(GF(int("".join(map(str, k2[i:i + 16])), 2))) for i in range(0, (degree + 1) * 16, 16)]
    k2_coefficients1 = [int(GF(int("".join(map(str, k2[i:i + 16])), 2))) for i in range(0, (degree + 1) * 16, 16)]
    print("k2_coefficients: ", k2_coefficients)
    print("---------------------------------------------------------------------------------------------")
    print("k2_coefficients1: ", k2_coefficients1)

    # Convert k1 binary partitions to field elements
    k1_points = [GF(int("".join(map(str, part)), 2)) for part in k1_partitions]
    
    # Evaluate polynomial at each point in k1
    results = []
    for x in k1_points:
        # Compute y = f(x) = a0 + a1*x + a2*x^2 + ... + ad*x^d
        y = GF(0)  # Start with zero in GF(2^16+1)
        for i, coeff in enumerate(k2_coefficients):
            y += coeff * (x**i)  # Perform addition and multiplication in GF(2^16)
        results.append((x, (GF(y))))
    
    return results, k2_coefficients

def decode_fuzzy_vault(vault, query_feature_vector, transformed_iriscode,coef_to_match, degree=7, part_size=16):
    """
    Decode a fuzzy vault using the query feature vector.
    Parameters:
        vault (list[tuple]): Vault points including chaff.
        query_feature_vector (list[int]): Binary query feature vector.
        transformed_iriscode (list[int]): Transformed iriscode.
        degree (int): Degree of the polynomial.
    Returns:
        bool: True if authentication succeeds, else False.
    """
    # Partition query feature vector into 6 components
    
    partitions = partition_feature_vector(query_feature_vector, part_size)
    print("len: ", len(transformed_iriscode), len(partitions))
    print("transformed_iriscode", transformed_iriscode)
    print("---------------------------------------------------------------------------------------------")
    
    print("query: ", partitions)
    print("---------------------------------------------------------------------------------------------")
    boolean = authenticate(partitions, transformed_iriscode,vault,degree,coef_to_match)
    if boolean: 
        return True
    else: 
        return False

def authenticate(query_iriscode, transformed_iriscode, vault, degree,coef_to_match):
    """
    Perform authentication using the query iriscode and the transformed iriscode.
    Parameters:
        query_iriscode (list[int]): Query iriscode (binary vector).
        transformed_iriscode (list[list[int]]): Transformed iriscode template.
        vault (list[tuple]): Fuzzy vault (list of (x, y) points).
        degree (int): Degree of the polynomial.
        bch_params (tuple): BCH parameters (n, k, t).
    Returns:
        bool: Authentication success or failure.
    """
# use bch decoding here . Currently removed for debugging purposes. 
    # Step 1: Inverse salting transformation
    # corrupted_codewords = [
    #     xor_vectors(query_part, transformed_part)
    #     for query_part, transformed_part in zip(query_iriscode, transformed_iriscode)
    # ]

    k1_partitions = [
        GF(int("".join(map(str, xor_vectors(query_part, transformed_part))), 2))
        for query_part, transformed_part in zip(query_iriscode, transformed_iriscode)
    ]
    print("k1 obtained: ", k1_partitions)
    print("---------------------------------------------------------------------------------------------")
    # Step 2: BCH decoding to recover transformation key
  
    # print(corrupted_codewords[2])

    # i=0
    # for corrupted_codeword in corrupted_codewords:
    #     decoded_key = bch_decode(corrupted_codeword)
    #     print(decoded_key)
    #     if decoded_key is not None:
    #         recovered_k1_parts.append(decoded_key)
    #     else:
    #         print("herrrewfe")
    #         print(i)
    #         return False  # BCH decoding failed for one or more codewords
    #     i+=1
    # Combine recovered key components
   

    # Step 3: Retrieve genuine points from the vault
    valid_vault_points = [
        ((GF(x)), (GF(y))) for x, y in vault if GF(x) in k1_partitions
    ]
    print("vault: ", vault)
    print("---------------------------------------------------------------------------------------------")    
    print("valid_vault_points", valid_vault_points)
    print("---------------------------------------------------------------------------------------------")    
    # Step 4: Polynomial reconstruction
    print('dp1',degree+1)
    i = 0
    for assumed_points in combinations(valid_vault_points, degree + 1):
        i += 1
        # Attempt polynomial reconstruction
        assumed_points = [i for i in assumed_points]
        print('assumed points:', assumed_points)
        reconstructed_poly = lagrange_coefficients(assumed_points)
        print("Reconstructed Polynomial:", reconstructed_poly)
        if reconstructed_poly == coef_to_match:
            print(reconstructed_poly)
            print("--------------------------")
            return True  # Successfully reconstructed polynomial
        if i > 1:
            break # REMOVE LATER

    # Step 5: Verify polynomial and key
    
    print("No poly found with suitable coefficients")
    # Authentication successful
    return False  # Authentication failed

def lagrange_coefficients(points):
    """
    Compute the coefficients of the Lagrange interpolating polynomial over GF(2^16).
    
    Parameters:
    points (list of tuples): A list of (x, y) pairs representing known points.
    
    Returns:
    list: Coefficients of the interpolating polynomial, from the highest to lowest degree.
    """
    # FIELD = GF(2**16 + 1)
    n = len(points)
    p = GF(1)
    coefficients = [GF(0)] * n  # Initialize coefficients list for the polynomial

    for i in range(n):
        # Lagrange basis polynomial L_i(x)
        x_i, y_i = GF(points[i][0]), GF(points[i][1])
        basis_coeff = [GF(1)]  # Coefficients for the current basis polynomial L_i(x)
        
        for j in range(n):
            if i != j:
                x_j = GF(points[j][0])
                # Update basis_coeff for (x - x_j) / (x_i - x_j)
                basis_coeff = multiply_polynomials(
                    basis_coeff, [GF(-x_j), GF(1)]  # Representing (x - x_j)
                )
                denom = GF(x_i - x_j)
                basis_coeff = [coef / denom for coef in basis_coeff]
        
        # Multiply L_i(x) by y_i and add to the total polynomial
        for k in range(len(basis_coeff)):
            coefficients[k] += basis_coeff[k] * y_i
    

    return coefficients


def multiply_polynomials(p1, p2):
    """
    Multiply two polynomials represented as coefficient lists over GF(2^16).
    
    Parameters:
    p1 (list): Coefficients of the first polynomial (highest to lowest degree).
    p2 (list): Coefficients of the second polynomial (highest to lowest degree).
    
    Returns:
    list: Coefficients of the resulting polynomial.
    """
    # FIELD = GF(2**16 + 1 )
    result = [GF(0)] * (len(p1) + len(p2) - 1)
    for i in range(len(p1)):
        for j in range(len(p2)):
            result[i + j] += p1[i] * p2[j]
    return result


def evaluate_fuzzy_vault_system(dataset_path, max_dirs=72):
    subdirectories = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    successful_decodes = 0

    for idx, subdir in enumerate(subdirectories[:max_dirs]):
        images = sorted([os.path.join(subdir, img) for img in os.listdir(subdir)])
        if len(images) < 2:
            continue  # Skip if less than two images in the directory

        # Use the first image for encoding
        ref_image = images[0]
        vec = try_FAR.process_eye_image(ref_image)
        ref_feature_vector = combineVectorbits(vec)
        vault, poly_coeffs, transformed_iriscode = create_fuzzy_vault(ref_feature_vector)

        # Use the second image for decoding
        query_image = images[1]
        vec = try_FAR.process_eye_image(query_image)
        query_feature_vector = combineVectorbits(vec)
        success = decode_fuzzy_vault(vault, query_feature_vector, transformed_iriscode,poly_coeffs,7)
    

        if success :
            successful_decodes += 1

        print(f"Processed {idx + 1}/{max_dirs} directories. Successful decodes: {successful_decodes}")

    success_rate = (successful_decodes / max_dirs) * 100
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    return success_rate

# test_fuzzy_vault()

# GF = GF(2**16 + 1)

# k1= [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1]]
# k2= [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1], [1, 1, 0]]

