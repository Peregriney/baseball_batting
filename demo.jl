ENV["PLOTS_TEST"]=true
ENV["GKSwstype"]=100
using Random, DataFrames, Distributed, Base.Threads, CSV, StatsBase, Permutations, Plots, TOML

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
DEFAULT_SIMS = config["dp"]["DEFAULT_SIMS"]
PROB_EXPORT_STR = config["dp"]["PROB_EXPORT_STR"]
PLOTHIST_EXPORT_STR = config["dp"]["PLOTHIST_EXPORT_STR"]
PLOTDIFF_EXPORT_STR = config["dp"]["PLOTDIFF_EXPORT_STR"]


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
memo = Dict{State, Float64}()
h_memo = Dict{Tuple{Int, Int, Int}, Float64}()
g_memo = Dict{Tuple{Int, Int, Int}, Float64}()
probmemo = Dict{Int, Float64}()


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
    println("p(", r, ") is ", sum)
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


# at_bat(index) : returns the outcome for a single player's at bat
function at_bat(index)
    # Retrieve the player's data from the dataset using the index
    playersInfo = playersData[index, 1:8]
    # Extract the probabilities for each outcome for the player
    playerSO = playersInfo[2] # Strikeout probability
    playerOO = playersInfo[3] # Other out probability
    playerBB = playersInfo[4] # Walk probability
    player1B = playersInfo[5] # Single probability
    player2B = playersInfo[6] # Double probability
    player3B = playersInfo[7] # Triple probability
    playerHR = 1 - (playerSO + playerOO + playerBB + player1B + player2B + player3B) # Home run probability

    # Create a list of outcomes and their corresponding probabilities
    outcomes = ["SO", "OO", "BB", "1B", "2B", "3B", "HR"]
    probabilities = [playerSO, playerOO, playerBB, player1B, player2B, player3B, playerHR]

    # Randomly select an outcome based on the probabilities
    outcome = outcomes[StatsBase.sample(1:length(outcomes), Weights(probabilities))]

    return outcome
end

# BB_bases: Function to update the bases when a player gets a walk (BB)
function BB_bases(bases::Vector{Int64})
    # Check the status of the bases and update them if a walk occurs
    # The following checks and updates are performed according to baseball rules
    if bases[1:3] == [0,0,0]
        bases[1] = 1
    elseif bases[1:3] == [0,0,1]
        bases[1] = 1
        bases[3] = 1 
    elseif bases[1:3] == [1,0,0] || bases[1:3] == [0,1,0]
        bases[1] = 1
        bases[2] = 1
    elseif bases[1:3] == [1,1,0] || bases[1:3] ==  [1,0,1] || bases[1:3] == [0,1,1]
        bases[1] = 1
        bases[2] = 1
        bases[3] = 1
    elseif bases[1:3] == [1,1,1]
        bases[1] = 1
        bases[2] = 1
        bases[3] = 1
        bases[4] += 1 # Increment the score because the bases were loaded
    end

    return bases
end

# single_bases: Function to update the bases when a player hits a single (1B)
function single_bases(bases::Vector{Int64})
    # Add runs for players on third base ****corrected from original simulation
    bases[4] += bases[3]
    # Clear third base
    bases[3] = 0
    # If there was a player on second, move him to third **corrected from og
    if bases[2] == 1
        bases[3] = 1
        bases[2] = 0
    end
    # If there was a player on first, move him to second
    if bases[1] == 1
        bases[2] = 1
    end
    # The hitter takes first base
    bases[1] = 1

    return bases
end

# double_bases: Function to update the bases when a player hits a double (2B)
function double_bases(bases::Vector{Int64})
    # Add runs for players on second and third base
    bases[4] += bases[2] + bases[3]
    # The hitter takes second base
    bases[2] = 1
    # Clear third base
    bases[3] = 0
    # If there was a player on first, move him to third
    if bases[1] == 1
        bases[3] = 1
        bases[1] = 0
    end

    return bases
end

# triple_bases: Function to update the bases when a player hits a triple (3B)
function triple_bases(bases::Vector{Int64})
    # Add runs for players on all bases
    bases[4] += bases[1] + bases[2] + bases[3]
    # The hitter takes third base
    bases[3] = 1
    # Clear first and second base
    bases[2] = 0
    bases[1] = 0

    return bases
end

# HR_bases: Function to update the bases when a player hits a home run (HR)
function HR_bases(bases::Vector{Int64})
    # Add runs for all players and the hitter
    bases[4] += bases[1] + bases[2] + bases[3] + 1
    # Clear all bases
    bases[3] = 0
    bases[2] = 0
    bases[1] = 0

    return bases
end

## game_outcome(lineup) : returns the number of runs scored from a given lineup, simulating a game
function game_outcome(lineup)
    # Initialize game variables
    inning = 0 # Inning counter
    runs = 0 # Total runs scored
    current_position = 0 # Current position in the batting lineup
    bases = [0, 0, 0, 0] # Bases array, where the 4th index holds the score
    sequence = [] # To record the sequence of plays

    # Loop through each inning
    while inning < 9
        inning += 1 # Increment inning
        outs = 0 # Reset outs at the start of each inning
        bases = [0,0,0,bases[4]] # Reset bases but keep the score

        # Loop until there are 3 outs
        while outs < 3
            current_position += 1 # Move to the next player in the lineup

            # Reset the lineup if we've gone through all players
            if current_position == 10
                current_position = 1
            end

            # Determine the outcome of the current player's at bat
            outcome = at_bat(lineup[current_position])
            # Record the outcome in the sequence
            push!(sequence,outcome)

            # Handle the outcome to update the number of outs or the bases
            if outcome == "SO" || outcome == "OO"
                outs += 1 # Increment outs for a strikeout or other out
            elseif outcome == "BB"
                bases = BB_bases(bases) # Update bases for a walk
            elseif outcome == "1B"
                bases = single_bases(bases) # Update bases for a single
            elseif outcome == "2B"
                bases = double_bases(bases) # Update bases for a double
            elseif outcome == "3B"
                bases = triple_bases(bases) # Update bases for a triple
            elseif outcome == "HR"
                bases = HR_bases(bases) # Update bases for a home run
            end
        end
    end

    # Return the total score and the sequence of at-bat outcomes
    return bases[4], sequence
end

# average_score: Calculates the average score from simulating a number of games
function average_score(lineup, num_games)
    total_score = 0 # Sum of scores from all games
    score_square_sum = 0 # Sum of squares of scores for variance calculation
    max_score = -Inf # Track the maximum score
    min_score = Inf # Track the minimum score

    # Simulate each game and update scoring statistics
    for game in 1:num_games
        # Obtain the score for the current game
        score = game_outcome(lineup)[1]
        # Aggregate the total score
        total_score += score
        # Aggregate the square of the score
        score_square_sum += score^2
        # Update the maximum score if the current score is higher
        max_score = max(max_score, score)
        # Update the minimum score if the current score is lower
        min_score = min(min_score, score)
    end

    # Calculate the average score
    avg_score = total_score / num_games
    # Calculate the variance
    variance = (score_square_sum / num_games) - avg_score^2
    # Calculate the standard deviation
    std_dev = sqrt(variance)

    # Return the average score, standard deviation, max score, and min score
    return avg_score, std_dev, max_score, min_score
end

# Argument handling
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
        lineup = randperm(NUM_BATTERS)
        println("Lineup ", lineup)
    end
end

function parse_single(arg)
    try
        return parse(Int64, arg)
    catch e
        println("Error: Number of games must be positive number.")
    end
end

#Argument handling
numSims = DEFAULT_SIMS
lineup = b_range
if length(ARGS) <2 || length(ARGS) != NUM_BATTERS + 2
    println("Incorrect argument formatting. Defaulting to redsox_2023.csv, 1k simulated games, batting lineup initialized to random.") 
    lineup = randperm(NUM_BATTERS)
    playersData = CSV.read(DEFAULT_STR, DataFrame)
    println("Lineup ", lineup)

elseif length(ARGS) == NUM_BATTERS + 2
    numSims = parse_single(ARGS[2])
    lineup = parse_args(ARGS[3:NUM_BATTERS+2])
    println("Accepted batting lineup ", lineup)
end

function Batter(idx::Int)::Int
    return lineup[idx]
end


# Begin main code -- populate memo, compute expected runs, run simulation, create plots
println("populating memo")
@time populateMemo()
println("DP: calculating expected runs")

@time e = expectedRuns(MAX_R,NUM_INNINGS)
println(e)

println()
println("Simulation: running simulated games to generate distribution of runs")

@time avg, stdev, maxx, minn = average_score(lineup, numSims)

# Create DataFrame with Metrics as Rows and DP & Simulated as Columns
metrics = ["Expected Number (Mean)", "Standard Deviation", "Minimum", "Maximum"]
dp_values = [expected_number, std_dev, min_val, max_val]
simulated_values = [avg, stdev, minn, maxx]

df = DataFrame(
    Metric = metrics,
    DP = dp_values,
    Simulated = simulated_values
)

# Display the DataFrame
println("Comparison of DP vs. Simulated Summary Stats:")
println(df)
println()

# average_score: Calculates the average score from simulating a number of games
function get_sim(lineup, num_games)
    total_score = 0 # Sum of scores from all games
    score_square_sum = 0 # Sum of squares of scores for variance calculation
    max_score = -Inf # Track the maximum score
    min_score = Inf # Track the minimum score

    scores = Int[]

    # Simulate each game and update scoring statistics
    for game in 1:num_games
        # Obtain the score for the current game
        score = game_outcome(lineup)[1]
        push!(scores, score)
        # Aggregate the total score
        total_score += score
        # Aggregate the square of the score
        score_square_sum += score^2
        # Update the maximum score if the current score is higher
        max_score = max(max_score, score)
        # Update the minimum score if the current score is lower
        min_score = min(min_score, score)
    end

    return scores, total_score/num_games
end


sim_scores, sim_avg = get_sim(lineup, numSims)

# Retrieves counts from simulated data
function count_occurrences(scores::Vector{Int})
    # Create a dictionary to store counts
    counts = Dict{Int, Int}()
    
    # Count occurrences of each element in scores
    for score in scores
        counts[score] = get(counts, score, 0) + 1
    end
    
    # Find the maximum value in scores to determine the size of the result array
    max_score = maximum(scores)
    
    # Create the result array with counts of distinct elements
    result = zeros(Int, max_score + 1)  # Array to hold counts from 0 to max_score
    
    # Populate the result array with counts
    for (key, count) in counts
        result[key + 1] = count
    end
    
    return result
end

sim_count_array = count_occurrences(sim_scores) ./ numSims


function sorted_values_by_keys(dict::Dict{Int, Float64})
    # Get the sorted keys
    sorted_keys = sort(collect(keys(dict)))
    
    # Get the values sorted by their keys
    sorted_values = [dict[key] for key in sorted_keys]
    
    return sorted_values
end

sorted_values = sorted_values_by_keys(probmemo)
n = length(sim_count_array)

plot(bar(collect(0:n-1),[sorted_values[1:n] sim_count_array ], label=["DP" "sim"], alpha=[1.0 0.5]),xlabel="Score (# of runs)", ylabel="Frequency (# of games)", title="Histogram for DP vs. Simulation", legend=true)
savefig(PLOTHIST_EXPORT_STR)

sliced_longer_array = sorted_values[1:n]
difference = sliced_longer_array .- sim_count_array 
# Plot the difference by index
plot(difference .* 100, label="Difference", xlabel="Index", ylabel="Difference (DP P(r) % - Sim Pr(r) %)", title="Element-wise % Difference of DP - Sim", xticks = 0:1:n, ylims=(-5,5))
savefig(PLOTDIFF_EXPORT_STR)


# Compute aggregated statistics for comparing dist from DP and Simulation
sum_abs_diff = sum(abs.(difference))
sum_squared_diff = sum(difference .^ 2)
correlation = cor(sorted_values[1:n], sim_count_array )
chi_square_stat = sum((difference .^ 2) ./ sorted_values[1:n])


# Create DataFrame with Metrics as Rows and DP & Simulated as Columns
metrics = ["Sum abs dif", "Sum sq dif", "Corr", "Chi-square"]
met_vals = [sum_abs_diff, sum_squared_diff, correlation,chi_square_stat]

df_metric = DataFrame(
    Metric = metrics,
    Value = met_vals,
)

# Display the DataFrame
println("Comparison of DP vs. Simulated histograms:")
println(df_metric)
println()


df_r = DataFrame(
  R = collect(0:n-1),
  Difference = difference .*100
)
println("Table of difference in Pr(r) (%) for DP - sim")
println(df_r)
println()

# Display the Dataframe comparing Pr(r)
df_rs = DataFrame(
  R = collect(0:n-1),
  DP = sorted_values[1:n].*100,
  Sim = sim_count_array.*100
)
println("Table of computed Pr(r) (%) for DP and Sim")
println(df_rs)
println()


