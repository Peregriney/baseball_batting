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

# Compute the number of runs scored in a state
function runsScored(state::State)::Float64
    b, j, outs, base1, base2, base3 = state.b, state.j, state.outs, state.base1, state.base2, state.base3
    return j - 3 - base1 - base2 - base3
end

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
    global penultimate_index

    bat = Batter(i)
    if bat == penultimate_index
      return 9 * playersData[bat, 5] - sum(playersData[x, 5] for x in 1:8)
    end
    return playersData[bat,5]
end

function Double(i::Int)::Float64
    global penultimate_index

    bat = Batter(i)
    if bat == penultimate_index
      return 9 * playersData[bat, 6] - sum(playersData[x, 6] for x in 1:8)
    end
    return playersData[bat,6]
end

function Triple(i::Int)::Float64
    global penultimate_index

    bat = Batter(i)
    if bat == penultimate_index
      return 9 * playersData[bat, 7] - sum(playersData[x, 7] for x in 1:8)
    end
    return playersData[bat,7]
end

function HomeRun(i::Int)::Float64
    global penultimate_index

    bat = Batter(i)
    if bat == penultimate_index
      return 9 * playersData[bat, 8] - sum(playersData[x, 8] for x in 1:8)
    end
    return playersData[bat,8]
end

function Walk(i::Int)::Float64
    global penultimate_index

    bat = Batter(i)
    if bat == penultimate_index
      return 9 * playersData[bat, 4] - sum(playersData[x, 4] for x in 1:8)
    end
    return playersData[bat,4]
end

function OutGet(i::Int)::Float64
    global penultimate_index

    bat = Batter(i)
    if bat == penultimate_index
      return 9 * (playersData[bat, 2]+playersData[bat,3])- sum((playersData[x, 2]+playersData[x, 3]) for x in 1:8)
    end
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


# Read existing lineups from the CSV into a set
function read_existing_lineups(filename)
    df = CSV.read(filename, DataFrame)
    return Set(df.Lineup)  # Convert the lineup column to a Set for quick lookup
end

ludf = CSV.read("output.csv", DataFrame)
ludf.score2 = Vector{Float64}(undef, nrow(ludf))  # Initialize with `undef` values
ludf.expectedRs = Vector{String}(undef,nrow(ludf))
ludf.probRs = Vector{String}(undef,nrow(ludf))
seenLineups = Dict{String, Float64}()
# Get the index of the penultimate row
global penultimate_index = nrow(playersData) - 1

# Loop through each permutation
for i in 1:nrow(ludf)
    row = ludf[i,:]
    lineup_str = row.Lineup

    global lineup
    lineup_str = strip(lineup_str, ['[', ']'])

    if haskey(seenLineups, lineup_str)
        # Use the cached value
        lineupval, k, v = seenLineups[lineup_str]
        ludf.score2[i] = lineupval
        ludf.expectedRs[i] = k
        ludf.probRs[i] = v

        println("Lineup already seen")
    else
        
        # Parse the Lineup string manually
        try
            global penultimate_index
            
            # Remove single quotes and split by comma
            elements = split(replace(lineup_str, "'" => ""))
            elements = [replace(el, "," => "") for el in elements]

            # Remove any extra spaces and convert to integers
            lineup = [parse(Int, strip(el)) for el in elements]            
            # Replace the last element of the lineup with the penultimate index
            lineup[end] = penultimate_index
            
        catch e
            println("Error parsing lineup string: ", lineup_str)
            println("Error message: ", e)
            continue
        end

        print(lineup)
        if length(lineup) == 9
          # Clear memo arrays before each computation
          clearMemos()
          
          println("Processing lineup: ", lineup_str)
          
          # Time and compute results
          populateMemo()
          eearly = expectedRuns(MAX_R, NUM_INNINGS - 1)
          efull = expectedRuns(MAX_R, NUM_INNINGS)
          e = 0.25 * eearly + 0.75 * efull
        
          ludf.dp[i] = e
          ludf.expectedRs[i] = keys(probmemo)
          ludf.probRs[i] = values(probmemo)
          seenLineups[lineup_str] = (e, keys(probmemo), values(probmemo))
        end
    end
end
println(ludf.score2)

CSV.write("v2_probs.csv", ludf)

