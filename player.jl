using CSV, DataFrames, Distributed, Base.Threads, Random, TOML, Combinatorics

# Define the state space
struct State
    b::Int
    j::Int
    outs::Int
    base1::Int
    base2::Int
    base3::Int
end

#Define memoization dictionaries
global memo = Dict{State, Float64}()
global h_memo = Dict{Tuple{Int, Int, Int}, Float64}()
global g_memo = Dict{Tuple{Int, Int, Int}, Float64}()
global probmemo = Dict{Int, Float64}()
global batted_in_memo = Dict{Tuple,Float64}()

config = TOML.parsefile("config.toml")

#Define variable ranges from config
b_range = config["dp"]["b_range"]
j_range = config["dp"]["j_range"]
outs_range = config["dp"]["outs_range"]
bases_range = config["dp"]["bases_range"]
NUM_BATTERS = config["dp"]["NUM_BATTERS"]
DEFAULT_STR = config["dp"]["DEFAULT_STR"]
NUM_INNINGS = config["dp"]["NUM_INNINGS"]
MAX_R = config["dp"]["MAX_R"]
PROB_EXPORT_STR = config["dp"]["PROB_EXPORT_STR"]
F_EXPORT_STR = config["dp"]["F_EXPORT_STR"]
G_EXPORT_STR = config["dp"]["G_EXPORT_STR"]
H_EXPORT_STR = config["dp"]["H_EXPORT_STR"]
PERMUTATION_EXPORT_STR = config["dp"]["PERMUTATION_EXPORT_STR"]


# f performs memoized recursion for probability of reaching states in an inning
function f(state::State)::Float64

    # Check if the value has already been computed
    if haskey(memo, state)
        return memo[state]
    end

    #Otherwise, compute the value
    b, j, outs, base1, base2, base3 = state.b, state.j, state.outs, state.base1, state.base2, state.base3
    # Base case 1
    if j == 0 && outs == 0 && base1 == 0 && base2 == 0 && base3 == 0 
        value = 1.0
        memo[state] = value
        return value
    end

    # Base case 2
    if j == 0
        value = 0.0
        memo[state] = value
        return value
    end

    # Base case 3
    if outs == -1 
        value = 0.0
        memo[state] = value
        return value
    end

    b_next = Batters(b,j)
    if outs == 3
        value = f(State(b, j - 1, 2, base1, base2, base3)) * OutGet(b_next)

    elseif base1 == 0 && base2 == 0 && base3 == 0
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                HomeRun(b_next) * sum(f(State(b, j - 1, outs, b1, b2, b3)) for b1 in [0, 1], b2 in [0, 1], b3 in [0, 1])
    elseif base1 == 1 && base2 == 0 && base3 == 0
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs, 0, 0, 0)) * Walk(b_next) +
                f(State(b, j - 1, outs, 0, 0, 0)) * Single(b_next) +
                f(State(b, j - 1, outs, 0, 0, 1)) * Single(b_next)
    elseif base1 == 0 && base2 == 1 && base3 == 0
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs, 0, 0, 0)) * Double(b_next) +
                f(State(b, j - 1, outs, 0, 1, 0)) * Double(b_next) +
                f(State(b, j - 1, outs, 0, 1, 1)) * Double(b_next) +
                f(State(b, j - 1, outs, 0, 0, 1)) * Double(b_next)
    elseif base1 == 0 && base2 == 0 && base3 == 1 #modified from doc
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                Triple(b_next) * sum(f(State(b, j - 1, outs, b1, b2, b3)) for b1 in [0, 1], b2 in [0, 1], b3 in [0, 1])
    elseif base1 == 1 && base2 == 1 && base3 == 0
        value = f(State(b,j-1,outs-1,base1,base2,base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs, 1, 0, 0)) * Walk(b_next) +
                f(State(b, j - 1, outs, 0, 1, 0)) * Walk(b_next) +
                f(State(b, j - 1, outs, 1, 0, 0)) * Single(b_next) +
                f(State(b, j - 1, outs, 1, 0, 1)) * Single(b_next)
    elseif base1 == 1 && base2 == 0 && base3 == 1
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs, 0, 0, 1)) * Walk(b_next) +
                f(State(b, j - 1, outs, 0, 1, 0)) * Single(b_next) +
                f(State(b, j - 1, outs, 0, 1, 1)) * Single(b_next)
    elseif base1 == 0 && base2 == 1 && base3 == 1
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs, 1, 0, 0)) * Double(b_next) +
                f(State(b, j - 1, outs, 1, 1, 0)) * Double(b_next) +
                f(State(b, j - 1, outs, 1, 0, 1)) * Double(b_next) +
                f(State(b, j - 1, outs, 1, 1, 1)) * Double(b_next)
    elseif base1 == 1 && base2 == 1 && base3 == 1
        value = f(State(b,j-1,outs-1,base1,base2,base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs, 1, 1, 0)) * Walk(b_next) +
                f(State(b, j - 1, outs, 1, 0, 1)) * Walk(b_next) +
                f(State(b, j - 1, outs, 0, 1, 1)) * Walk(b_next) +
                f(State(b, j - 1, outs, 1, 1, 1)) * Walk(b_next) +
                f(State(b, j - 1, outs, 1, 1, 0)) * Single(b_next) +
                f(State(b, j - 1, outs, 1, 1, 1)) * Single(b_next)
    end
    
    memo[state] = value
    #Probability to reach a given state of where first batter is Player(b), j batters have come to bat, and specified outs and bases
    return value
end


#PSEUDOCODE:
#recall: f[b_i, j, OUTS, bs1, bs2, bs3] = probability you reach that state. 

#FROM any state, the RBI for player b' is 

#battedIn() function computes RBI for player b' from all states?. get TOTAL expected batted-in for player bprime
#takes a specific b_prime [mod DIFFERENCE between b_i and j] and the original b_i
   # SUM for all states where b_prime = Batters(b,j) (get TOTAL expected batted-in for player bprime)
#this is the same for all innings because it relies solely on f --> solely on the within-inning probabilities of reaching a certain outcome, given starting conditions.

#getNewBi()
#so for inning 1: to get RBI in ING1 for players i in 1:9. battedIn(1, i=1) : battedIn(1,i=9)

#what's difficult is inning 2, 3, 4... 
#call ing1 = [p1END, p2END, ... p9END]
#call ing2 = [p1start_ing2, ... p9start_ing2]

#for ing2, iterate all Players we're interested in (i = 1:9) computing the END_ing2 probability for, then iterate all potential Starting players x = 1:9 
#[[battedIn(start=x,end=i) * ProbInningStartedOn(X) ]] => sum for each player i 1:9 in fullrbi


#computeNextIng() => this function takes in the start probabilities of current inning and returns start prob of NEXT inning
#for player b' we're interestede, in 1:9: 
#




function getNewBi(inning)::Dict
    fullrbi = Dict(i => 0.0 for i in 1:9)

    #compute for inning 1 probabilities of ending on any give player i
    sumstore = Dict(i => 0.0 for i in 1:9)
    totalsum = 0.0
    for (state, value) in memo
        if state.b == 1 && state.outs == 3 
            totalsum += value
            sumstore[Batters(state.b,state.j)] += value
        end
    end
    ing1 = Dict(k => v / totalsum for (k, v) in sumstore)
    ing2 = Dict(n => ing1[(n - 1) % 9 + 1] for n in 1:9)
    println(sum(values(ing2)))
    #where ing2 represents the likelihood of ing2 starting with each of these players
    for i in 1:9
      fullrbi[i] += battedIn(1,i)
    end
    #this populates fullrbi with the RBIs from inning 1. 

    println("RBI after ing1")
    println(fullrbi)
      

    #innings 2 through 9 are trickier

    #MAX_INNINGS = rand() < 0.25 ? 8 : 9
    for ing in 2:9
        
      for i in 1:9
        for n in 1:9
          if ing == 9
            fullrbi[i] += .75 * battedIn(n,i) * ing2[n] ###flag
          else 
            fullrbi[i] += battedIn(n,i) * ing2[n] ###flag but checked tbh
          end
    
        end
      end
      println("RBIs after inning ", ing)
      println(fullrbi)
      ingnext = computeNextIng(ing2)
      #intermediate: ingnext represents probability of ING=ing ending on 1:9
      ing2 = Dict(n => ingnext[(n - 1) % 9 + 1] for n in 1:9)
      #ing2 computes probability of ing_next STARTING with one of 1:9

      println("Probabilities of next inning starting on players 1thru9 ")
      println(ing2)
      println(sum(values(ing2)))
    end
    return fullrbi
end

function computeNextIng(ing2::Dict{Int, Float64})::Dict{Int,Float64}
    ing3 = Dict{Int,Float64}()
    #initialize the ending probs for ing2.

    #ing 2 is chance you start on each of p1, p2, ... p9. 
    #for playeri in 1:9
        #chance that you END on playeri (example i=1) is the
            #SUM over n in 1:9 where n is wrapper: sum the
            #chance you started at p1 * probability you score mod wrap 9 + ing2[p2]*sumStore[scoremod8] + ing2[p3]*sumStore[scoremod7]
            # SUM[n in 1:9] ing2[(i+n) mod 9] * sumStore[n]

        #chance that you END on player i is the 
            #SUM: 
            # startProb[i] * modWrapProb[9]
            #startProb[i+1] * modWrapProb[1]
            #startProb[i+2] * modWrapProb[2] ... = SIGMA_n ing2[(i+n - 1) % 9] adj for 0-index, * sumstore[n]
    for i in 1:9
        sumstore = Dict(i => 0.0 for i in 1:9)
        totalsum = 0.0
        for (state, value) in memo
            if state.b == i && state.outs == 3 
                totalsum += value
                sumstore[Batters(state.b,state.j)] += value
            end
        end
        sumstore = Dict(k => v / totalsum for (k, v) in sumstore)
        for n in 1:9
            ing3[i] = get(ing3, i, 0.0) + (ing2[((i+n−1−1)%9)+1] * sumstore[n])
        end
    end
    return ing3
end


# Compute the number of runs scored in a state
function runsScored(state::State)::Float64
    b, j, outs, base1, base2, base3 = state.b, state.j, state.outs, state.base1, state.base2, state.base3
    return j - 3 - base1 - base2 - base3
end

#expectation of player bprime runs batted in 
function battedIn(b_i, b_prime::Int)::Float64
    key = (b_i, b_prime)
    if haskey(batted_in_memo, key)
        return batted_in_memo[key]
    end


    #SUM for all states where b_prime = Batters(b,j) (get TOTAL expected batted-in for player bprime)
    sumstore = 0
    #for b_i in b_range
    for (state, value) in memo
        
        if state.b == b_i && Batters(state.b,state.j) == b_prime && state.outs <3
            if state.base1 == 0 && state.base2 == 0 && state.base3 == 0
              sumstore += HomeRun(b_prime) * value
            end
            if state.base1 == 1 && state.base2 == 0 && state.base3 == 0
              sumstore += (2*HomeRun(b_prime) + 1*Triple(b_prime)) * value
            end
            if state.base1 == 1 && state.base2 == 1 && state.base3 == 0
              sumstore += (3*HomeRun(b_prime) + 2*Triple(b_prime) + 1*Double(b_prime)) * value
            end
            if state.base1 == 1 && state.base2 == 0 && state.base3 == 1
              sumstore += (3*HomeRun(b_prime) + 2*Triple(b_prime) + 1*Double(b_prime) + 1*Single(b_prime)) * value
            end
            if state.base1 == 1 && state.base2 == 1 && state.base3 == 1
              sumstore += (4*HomeRun(b_prime) + 3*Triple(b_prime) + 2*Double(b_prime) + 1*Single(b_prime) + 1*Walk(b_prime)) * value
            end
            if state.base1 == 0 && state.base2 == 1 && state.base3 == 0
              sumstore += (2*HomeRun(b_prime) + 1*Triple(b_prime) + 1*Double(b_prime)) * value
            end
            if state.base1 == 0 && state.base2 == 0 && state.base3 == 1
              sumstore += (2*HomeRun(b_prime) + 1*Triple(b_prime) + 1*Double(b_prime) + 1*Single(b_prime)) * value
            end
            if state.base1 == 0 && state.base2 == 1 && state.base3 == 1
              sumstore += (3*HomeRun(b_prime) + 2*Triple(b_prime) + 2*Double(b_prime) + 1*Single(b_prime)) * value
            end
                        
        #  sumstore += sum((runsScored(state)- runsScored(State(state.b,state.j+1,outs,b1,b2,b3))) * memo[State(state.b,state.j,outs,b1,b2,b3)] for outs in [0,1,2], b1 in [0,1], b2 in [0,1], b3 in [0,1]) 
        end
    end
#    end

    batted_in_memo[key] = sumstore
    return sumstore
end


#sum(f(State(b, j - 1, outs, b1, b2, b3)) for b1 in [0, 1], b2 in [0, 1], b3 in [0, 1])

#sum of all states in f-memo where outs == 3 AND b=b and j = b_prime and r = runsScored(state)
function g(b::Int, b_prime::Int, r::Int)::Float64
  key = (b, b_prime, r)
  if haskey(g_memo, key)
      return g_memo[key]
  end

  sumstore = 0
    for (state, value) in memo
        if state.outs == 3 && state.b == b && Batters(state.b,state.j) == b_prime && r == runsScored(state)
            sumstore += value
        end
    end
  g_memo[key] = sumstore
  return sumstore
end

#Compute probability of getting score r summing aggregating over innings
function h_parallel(i::Int, b_prime::Int, r::Int)::Float64
    key = (i, b_prime, r)
    if haskey(h_memo, key)
        return h_memo[key]
    end

    if i == 1
        result = g(1, b_prime, r)
    else
        result = 0.0
        for b in b_range
            local sum_b = 0.0
            for r_prime in 0:r
                sum_b += h_parallel(i-1, b, r_prime) * g(Next(b), b_prime, r-r_prime)
            end
            result += sum_b
        end
    end

    h_memo[key] = result
    return result
end


#Compute overall probability of reaching state r in one game
function prob(r::Int, INGS::Int)::Float64
    sum = 0.0
    for b in b_range
        sum += h_parallel(INGS,b,r)
    end
    #println("p(", r, ") is ", sum)
    probmemo[r] = sum
    return sum
end

#Compute expected runs or expected score of one game
function expectedRuns(rmax::Int, INGS::Int)::Float64
    sum = Atomic{Float64}(0.0)  # Use an atomic variable for thread-safe summation

    @threads for r in 0:rmax
        Threads.atomic_add!(sum, r * prob(r, INGS))  # Safely add to the atomic sum
    end

    return sum[]
end

#Populate all entries in memo: b can be 1:9, j can be 1:27, outs can be 0:3, b1,b2,b3 can all be 0/1
function populateMemo()
    for b in b_range
        for j in j_range
            for outs in outs_range
                for base1 in bases_range
                    for base2 in bases_range
                        for base3 in bases_range
                            f(State(b,j,outs,base1,base2,base3))
                        end
                    end
                end
            end
        end
    end
end

# Define the functions for player statistics --> all of these need to retrieve from some array of values for each of the 9players
function Single(i::Int)::Float64
    bat = Batter(i)
    return playersData[bat,5]
end

function Double(i::Int)::Float64
    bat = Batter(i)
    return playersData[bat,6]
end

function Triple(i::Int)::Float64
    bat = Batter(i)
    return playersData[bat,7]
end

function HomeRun(i::Int)::Float64
    bat = Batter(i)
    return playersData[bat,8]
end

function Walk(i::Int)::Float64
    bat = Batter(i)
    return playersData[bat,4]
end

function OutGet(i::Int)::Float64
    bat = Batter(i)
    return playersData[bat,2] + playersData[bat,3]
end

# Define the function for the next batter
function Next(b::Int)::Int
    return (b % NUM_BATTERS) + 1
end

# Define the function for the j-th batter in an inning
function Batters(b::Int, j::Int)::Int
    return (b + j - 1) % NUM_BATTERS + 1
end

## Function to check if the probabilities in each row sum to 1
function check_probabilities(playersData)
    for i in 1:nrow(playersData)
        row = playersData[i, :]
        probability_sum = sum(row[2:8]) 
        if probability_sum < 0.99 || probability_sum > 1.01  # Checking if sum ~= 1
            println("Error: Probabilities in row ", i, " do not sum to 1. Sum is: ", probability_sum)
        else
            print()
        end
    end
    println("Player batting probabilities all sum to 1")
end

function try_read()
    try 
        csv_str = ARGS[1]
        data_csv = CSV.read(csv_str, DataFrame)
        return data_csv
    catch e
        println("Error: first argument must be CSV file, could not read. Format as [pathname.csv] to command-line. Defaulting to redsox_2023.csv")
        data_csv = CSV.read(DEFAULT_STR,DataFrame)
        return data_csv
    end
end

playersData = CSV.read(DEFAULT_STR, DataFrame)

# Read probabilities and check that each player's stats sum to 1
if length(ARGS) <1
    println("No filename specified, defaulting to redsox_2023.csv")
    playersData = CSV.read(DEFAULT_STR, DataFrame)
else
    playersData = try_read()
end

check_probabilities(playersData)

function parse_args(args)
    try
        return [parse(Int64, arg) for arg in args]
    catch e
        println("Error: All arguments must be numbers. Setting random lineup.")
        global lineup = randperm(NUM_BATTERS)
        println("lineup", lineup)
    end
end

# Randomly initialize batting order (lineup) if none given
if length(ARGS) <2 || length(ARGS) != NUM_BATTERS+1
    println("batting lineup initialized to random. No lineup given") 
    global lineup = randperm(NUM_BATTERS)
    println("lineup", lineup)
elseif length(ARGS) == NUM_BATTERS + 1
    global lineup = parse_args(ARGS[2:NUM_BATTERS+1])
    println("accepted batting lineup", lineup)
end

function Batter(idx::Int)::Int
    global lineup
    return lineup[idx]
end

function clearMemos()
  global memo = Dict{State, Float64}()
  global h_memo = Dict{Tuple{Int, Int, Int}, Float64}()
  global g_memo = Dict{Tuple{Int, Int, Int}, Float64}()
  global probmemo = Dict{Int, Float64}()

end


lineups = permutations(1:NUM_BATTERS)

println("Processing f memo")
@time populateMemo()


println("RBIs by Player: ")
fullrbi = getNewBi(1)
for b in b_range
  println(fullrbi[b])
end

println("sum of RBIs for all players")
println(sum(values(fullrbi)))

