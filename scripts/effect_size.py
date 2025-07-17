import sys
import numpy as np
    

def effect_size_mann_whitney(z, n):
    return z / np.sqrt(n)

def main():
    print("Choose the method for effect size calculation: [mann-whitney [m], lmm]")
    method = input().strip().lower()
    if method not in ["mann-whitney", "m", "lmm"]:
        print("Invalid method. Please choose 'mann-whitney' or 'lmm'.")
        sys.exit(1)
        
    if method == "mann-whitney" or method == "m":
        z = float(input("Enter Z value from SPSS: "))
        n = int(input("Enter total sample size (n1 + n2): "))
        r = effect_size_mann_whitney(z, n)
        print(f"\nEffect size r = {abs(r):.3f}")
        print("Interpretation:", end=" ")
        if abs(r) < 0.1:
            print("Very small")
        elif abs(r) < 0.3:
            print("Small")
        elif abs(r) < 0.5:
            print("Medium")
        else:
            print("Large")
            
    elif method == "lmm":
        print("LMM effect size calculation is not implemented yet.")
        

if __name__ == "__main__":
    main()
