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
        if base1 + base2 + base3 == 0
          value = f(State(b, j - 1, 2, base1, base2, base3)) * OutGet(b_next) #+ 
          #f(State(b, j, outs-1, 1, 0, 0)) * StealOut() + f(State(b, j, outs-1, 0, 1, 0)) * StealOut()

          
        else
          value = f(State(b, j - 1, 2, base1, base2, base3)) * OutGet(b_next) + 
          DoubleOut(b_next) * f(State(b, j-1, 1, base1, base2, base3))
            #any time that there is at least one batter on base. can get to 3 outs from 1 out if a doubleplay happens.
            #Use general GDP rate 

        end

    elseif base1 == 0 && base2 == 0 && base3 == 0
        if outs  == 2
            value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                HomeRun(b_next) * sum(f(State(b, j - 1, outs, b1, b2, b3)) for b1 in [0, 1], b2 in [0, 1], b3 in [0, 1]) + 
		            .3 * sum(f(State(b, j - 1, 0, b1, b2, b3)) for (b1,b2,b3) in [(1,0,0), (1,1,0)]) + 
                #case for conditional .25 when baseman on first
                DoubleOut(b_next) * sum(f(State(b, j - 1, 0, b1, b2, b3)) for (b1,b2,b3) in [(0,1,0), (0,1,1)]) +
                #case for other GIDP circumstances 
                f(State(b, j, outs-1, 1, 0, 0)) * StealOut() + f(State(b, j, outs-1, 0, 1, 0)) * StealOut() 
                + f(State(b, j, outs, 0, 0, 1)) * StealSuccess() + 
                f(State(b, j-1, outs-2, 0, 1, 0)) * OutGet(b_next) * StealOut() + 
                f(State(b, j-1, outs-2, 1, 0, 0)) * OutGet(b_next) * StealOut()


        else
            value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                HomeRun(b_next) * sum(f(State(b, j - 1, outs, b1, b2, b3)) for b1 in [0, 1], b2 in [0, 1], b3 in [0, 1]) + 
                f(State(b, j, outs-1, 1, 0, 0)) * StealOut() + f(State(b, j, outs-1, 0, 1, 0)) * StealOut() 
                


        end
    elseif base1 == 1 && base2 == 0 && base3 == 0
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs-2, 1,1,0)) * OutGet(b_next) * StealOut() + #modded
                f(State(b, j - 1, outs-1, 1,1,0)) * Single(b_next) * StealOut() + #modded
                f(State(b, j - 1, outs, 0, 0, 0)) * Walk(b_next) +
                f(State(b, j - 1, outs, 0, 0, 0)) * Single(b_next) +
                f(State(b, j - 1, outs, 0, 0, 1)) * Single(b_next) 
                + f(State(b, j-1, outs-1, 0, 1, 0)) * Single(b_next) * StealOut() + 
                f(State(b, j-1, outs, 0, 1, 0)) * Single(b_next) * StealSuccess() + 
                f(State(b, j-1, outs-1, 1, 0, 0)) * Single(b_next) * StealOut()


    elseif base1 == 0 && base2 == 1 && base3 == 0
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                f(State(b, j-1, outs-1, 1, 0, 0)) * OutGet(b_next) * StealSuccess() + 

                f(State(b, j - 1, outs, 0, 0, 0)) * Double(b_next) +
                f(State(b, j - 1, outs, 0, 1, 0)) * Double(b_next) +
                f(State(b, j - 1, outs, 0, 1, 1)) * Double(b_next) +
                f(State(b, j - 1, outs, 0, 0, 1)) * Double(b_next) +
                f(State(b, j, outs, 1, 0, 0)) * StealSuccess()  #modded steal
        
    elseif base1 == 0 && base2 == 0 && base3 == 1 #modified from doc
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                Triple(b_next) * sum(f(State(b, j - 1, outs, b1, b2, b3)) for b1 in [0, 1], b2 in [0, 1], b3 in [0, 1]) +
                f(State(b, j, outs, 0, 1, 0)) * StealSuccess() + #modded steal
                f(State(b, j-1, outs-1, 0, 1, 0)) * OutGet(b_next) * StealSuccess()
    elseif base1 == 1 && base2 == 1 && base3 == 0
        value = f(State(b,j-1,outs-1,base1,base2,base3)) * OutGet(b_next) +
                f(State(b, j-1, outs, 1, 1, 0)) * Single(b_next) * StealSuccess() + 
                f(State(b, j - 1, outs, 1, 0, 0)) * Walk(b_next) +
                f(State(b, j - 1, outs, 0, 1, 0)) * Walk(b_next) +
                f(State(b, j - 1, outs, 1, 0, 0)) * Single(b_next) +
                f(State(b, j - 1, outs, 1, 0, 1)) * Single(b_next)
    elseif base1 == 1 && base2 == 0 && base3 == 1
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs, 0, 0, 1)) * Walk(b_next) +
                f(State(b, j - 1, outs, 0, 1, 0)) * Single(b_next) +
                f(State(b, j - 1, outs, 0, 1, 1)) * Single(b_next) +
                f(State(b, j-1, outs, 1, 0, 0)) * Single(b_next) * StealSuccess() + 
                f(State(b, j, outs, 1, 1, 0)) * StealSuccess() #modded steal
        
    elseif base1 == 0 && base2 == 1 && base3 == 1
        value = f(State(b, j - 1, outs-1, base1, base2, base3)) * OutGet(b_next) +
                #f(State(b, j - 1, outs-1, 1,1,0)) * OutGet(b_next) * StealSuccess() +
                f(State(b, j - 1, outs, 1, 0, 0)) * Double(b_next) +
                f(State(b, j - 1, outs, 1, 1, 0)) * Double(b_next) +
                f(State(b, j - 1, outs, 1, 0, 1)) * Double(b_next) +
                f(State(b, j - 1, outs, 1, 1, 1)) * Double(b_next) +
                f(State(b, j, outs, 1, 0, 1)) * StealSuccess() #modded steal
                
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


function getOnBase(inning)::Tuple
   
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
    #where ing2 represents the likelihood of ing2 starting with each of these players

    leftOnBase, jmax = leftOn([1,0,0,0,0,0,0,0,0])
    #this populates the number of left-on-base batters from inning 1 and the 
    #average number of total batters that went up. 

    for ing in 2:9
        
    
      if ing == 9
        lob, jmax_ing = leftOn(ing2)
        leftOnBase += .75 * lob ###flag
        jmax += jmax_ing
      else 
        lob, jmax_ing = leftOn(ing2)
        leftOnBase += lob ###flag but checked tbh
        jmax += jmax_ing
      end    
    
      ingnext = computeNextIng(ing2)
      #intermediate: ingnext represents probability of ING=ing ending on 1:9
      ing2 = Dict(n => ingnext[(n - 1) % 9 + 1] for n in 1:9)
      #ing2 computes probability of ing_next STARTING with one of 1:9

    end
    return leftOnBase, jmax
end

function leftOn(inningprobs)
    totalsum = 0
    totalj = 0
    for b in 1:9

        sumstore = 0
        jstore = 0

        for (state, value) in memo
            if state.outs == 3 && state.b == b #((state.b + -1) % 9) + 1 == b
                sumstore += inningprobs[b] * value * (state.base1 + state.base2 + state.base3)
                
                jstore += inningprobs[b] * value * (state.j)
            end
        end
        totalsum += sumstore
        totalj += jstore
    
    end
    return totalsum, totalj
end

function computeNextIng(ing2::Dict{Int, Float64})::Dict{Int,Float64}
    ing3 = Dict{Int,Float64}()
    #initialize the ending probs for ing2.

    #ing 2 is chance you start on each of p1, p2, ... p9. 
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

#Compute expected runs or expected score of one game
function expectedRunsNormed(rmax::Int, INGS::Int)::Float64
    @threads for r in 0:rmax
        prob(r,INGS)
    end
    global probmemo
    total = sum(values(probmemo))
    normalized_probs = total != 0 ? Dict(k => v / total for (k, v) in probmemo) : Dict()
    probmemo = normalized_probs

    atsum = Atomic{Float64}(0.0)  # Use an atomic variable for thread-safe summation

    @threads for r in 0:rmax
        Threads.atomic_add!(atsum, r * probmemo[r])  # Safely add to the atomic sum
    end

    return atsum[]
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

function StealSuccess()
    global penultimate_index
    return playersData[penultimate_index, 9]
end

function StealOut()
    global penultimate_index
    return playersData[penultimate_index, 10]
end

function DoubleOut(i::Int)::Float64
    global penultimate_index

    bat = Batter(i)
    if bat == penultimate_index
      return 9 * playersData[bat, 11] - sum(playersData[x, 11] for x in 1:8)
    end
    return playersData[bat,11]
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
if length(ARGS) <2 || length(ARGS) != NUM_BATTERS+2
    println("batting lineup initialized to random. No lineup given") 
    global lineup = randperm(NUM_BATTERS)
    println("lineup", lineup)
elseif length(ARGS) == NUM_BATTERS + 2
    global lineup = parse_args(ARGS[2:NUM_BATTERS+1])
    println("accepted batting lineup", lineup)
    global team = ARGS[NUM_BATTERS+2]
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

global team
ludf = CSV.read("output.csv", DataFrame)
ludf.team = fill(team, nrow(ludf))

ludf.onbase = Vector{Float64}(undef, nrow(ludf))
ludf.j = Vector{Float64}(undef, nrow(ludf))

seenLineups = Dict{String, Tuple}()
# Get the index of the penultimate row
global penultimate_index = nrow(playersData)




for i in 1:30#nrow(ludf)

    row = ludf[i,:]
    lineup_str = row.Lineup

    global lineup
    global player
    global der
    lineup_str = strip(lineup_str, ['[', ']'])


    if haskey(seenLineups, lineup_str)
        # Use the cached value
        (avg_onbase, max_j) = seenLineups[lineup_str]
        ludf.onbase[i] = avg_onbase
        ludf.j[i] = max_j
        println("Lineup already seen")
    else
        
        # Parse the Lineup string manually
        try
            
            # Remove single quotes and split by comma
            elements = split(replace(lineup_str, "'" => ""))
            elements = [replace(el, "," => "") for el in elements]

            # Remove any extra spaces and convert to integers
            lineup = [parse(Int, strip(el)) for el in elements]
            
        catch e
            println("Error parsing lineup string: ", lineup_str)
            println("Error message: ", e)
            continue
        end

        print(lineup)
        if length(lineup) == 9
              clearMemos()
              println("Processing lineup: ", lineup_str)
              populateMemo()

              avg_onbase, max_j = getOnBase(1)
              println(avg_onbase)
              println(max_j)
              ludf.onbase[i] = avg_onbase
              ludf.j[i] = max_j
              seenLineups[lineup_str] = (avg_onbase, max_j)

        end
    end
end



CSV.write("v2_onbase_probs.csv", ludf,append=true)
