def ces_utility(x, y, delta=0.5):
    """
    Calculates the CES utility for bundle (x,y).
    Inputs:
    x: value good 1 (np.array)
    y: value good 2 (np.array)
    delta: elasticity of good 1 and good 2 (float)
    Output:
    u: utility value (np.array)
    """
    if delta >= 1:
        raise ValueError('Delta is not less than 1.')
    u = x**delta/delta + y**delta/delta
    return(u)

def main():
    while True: # pretend there is a long script here
        print(ces_utility(1, 2, delta=0.5))

if __name__ == '__main__':
    main()