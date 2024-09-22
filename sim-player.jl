ENV["PLOTS_TEST"]=true
ENV["GKSwstype"]=100

using Random
using DataFrames, CSV
using StatsBase
using Permutations
using Plots, TOML

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
function BB_bases(bases::Vector{Int64}, player, rs, rbi)
    # Check the status of the bases and update them if a walk occurs
    # The following checks and updates are performed according to baseball rules

    bases_bool = map(x -> x > 0 ? 1 : 0, bases)

    if bases_bool[1:3] == [0,0,0]
        bases[1] = player
    elseif bases_bool[1:3] == [0,0,1]
        bases[1] = player
        bases[3] = bases[3]
    elseif bases_bool[1:3] == [1,0,0] 
        bases[2] = bases[1]
        bases[1] = player
    elseif bases_bool[1:3] == [0,1,0]
        bases[1] = player
    elseif bases_bool[1:3] == [1,1,0] 
        bases[3] = bases[2]
        bases[2] = bases[1]
        bases[1] = player
    elseif bases_bool[1:3] ==  [1,0,1] 
        bases[2] = bases[1]
        bases[1] = player
    elseif bases_bool[1:3] == [0,1,1]
        bases[1] = player
    elseif bases_bool[1:3] == [1,1,1]
        rs[bases[3]] += 1
        rbi[player] += 1
        bases[3] = bases[2]
        bases[2] = bases[1]
        bases[1] = player
        bases[4] += 1 # Increment the score because the bases were loaded
    end

    return bases
end

# single_bases: Function to update the bases when a player hits a single (1B)
function single_bases(bases::Vector{Int64}, player, rs, rbi)
    # Add runs for players on third base ****corrected from original simulation
    non_zero_count = count(x -> x != 0, bases[3:3])
    bases[4] += non_zero_count

    if bases[3] > 0
      rs[bases[3]] += 1
    end
    rbi[player] += non_zero_count

    # Clear third base
    bases[3] = 0
    # If there was a player on second, move him to third **corrected from og
    if bases[2] > 0
        bases[3] = bases[2]
        bases[2] = 0
    end
    # If there was a player on first, move him to second
    if bases[1] > 0
        bases[2] = bases[1]
    end
    # The hitter takes first base
    bases[1] = player

    return bases
end

# double_bases: Function to update the bases when a player hits a double (2B)
function double_bases(bases::Vector{Int64}, player, rs, rbi)
    # Add runs for players on second and third base
    non_zero_count = count(x -> x != 0, bases[2:3])
    
    bases[4] += non_zero_count

    if bases[2] > 0
      rs[bases[2]] += 1
    end
    if bases[3] > 0
      rs[bases[3]] += 1
    end
    rbi[player] += non_zero_count

    # The hitter takes second base
    bases[2] = player
    # Clear third base
    bases[3] = 0
    # If there was a player on first, move him to third
    if bases[1] > 0
        bases[3] = bases[1]
        bases[1] = 0
    end

    return bases
end

# triple_bases: Function to update the bases when a player hits a triple (3B)
function triple_bases(bases::Vector{Int64}, player, rs, rbi)
    # Add runs for players on all bases
    non_zero_count = count(x -> x != 0, bases[1:3])
    bases[4] += non_zero_count
    if bases[1] > 0
      rs[bases[1]] += 1
    end
    if bases[2] > 0
      rs[bases[2]] += 1
    end
    if bases[3] > 0
      rs[bases[3]] += 1
    end
    rbi[player] += non_zero_count

    # The hitter takes third base
    bases[3] = player
    # Clear first and second base
    bases[2] = 0
    bases[1] = 0

    return bases#, rs, rbi
end

# HR_bases: Function to update the bases when a player hits a home run (HR)
function HR_bases(bases::Vector{Int64}, player, rs, rbi)
    # Add runs for all players and the hitter
    non_zero_count = count(x -> x != 0, bases[1:3])
    bases[4] += non_zero_count + 1
    if bases[1] > 0
      rs[bases[1]] += 1
    end
    if bases[2] > 0
      rs[bases[2]] += 1
    end
    if bases[3] > 0
      rs[bases[3]] += 1
    end
    rs[player] += 1
    rbi[player] += non_zero_count + 1

    # Clear all bases
    bases[3] = 0
    bases[2] = 0
    bases[1] = 0

    return bases#, rs, rbi
end

## game_outcome(lineup) : returns the number of runs scored from a given lineup, simulating a game
function game_outcome(lineup)
    # Initialize game variables
    inning = 0 # Inning counter
    runs = 0 # Total runs scored
    current_position = 0 # Current position in the batting lineup
    bases = [0, 0, 0, 0] # Bases array, where the 4th index holds the score
    sequence = [] # To record the sequence of plays

    rs = Dict(i => 0 for i in 0:9)
    rbi = Dict(i => 0 for i in 0:9)

    MAX_INNINGS = rand() < 0.25 ? 8 : 9
    # Loop through each inning
    while inning < MAX_INNINGS
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
                bases = BB_bases(bases, current_position,rs,rbi) # Update bases for a walk
            elseif outcome == "1B"
                bases = single_bases(bases, current_position,rs,rbi) # Update bases for a single
            elseif outcome == "2B"
                bases = double_bases(bases, current_position,rs,rbi) # Update bases for a double
            elseif outcome == "3B"
                bases = triple_bases(bases, current_position,rs,rbi) # Update bases for a triple
            elseif outcome == "HR"
                bases = HR_bases(bases, current_position,rs,rbi) # Update bases for a home run
            end
        end
    end

    # Return the total score and the sequence of at-bat outcomes
    return bases[4], sequence, rs, rbi
end

# average_score: Calculates the average score from simulating a number of games
function average_score(lineup, num_games)
    total_score = 0 # Sum of scores from all games
    score_square_sum = 0 # Sum of squares of scores for variance calculation
    max_score = -Inf # Track the maximum score
    min_score = Inf # Track the minimum score

    player1rs = 0
    player2rs = 0
    player3rs = 0
    player4rs = 0
    player5rs = 0
    player6rs = 0
    player7rs = 0
    player8rs = 0
    player9rs = 0
    player1rbi = 0
    player2rbi = 0
    player3rbi = 0
    player4rbi = 0
    player5rbi = 0
    player6rbi = 0
    player7rbi = 0
    player8rbi = 0
    player9rbi = 0
    player_rss = zeros(Float64, 9)
    player_rbis = zeros(Float64, 9)

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

        rs = game_outcome(lineup)[3]
        for n in 1:9
          player_rss[n] += rs[n]
        end

        rbi = game_outcome(lineup)[4]
        for n in 1:9
          player_rbis[n] += rbi[n]
        end
    end

    # Calculate the average score
    avg_score = total_score / num_games
    # Calculate the variance
    variance = (score_square_sum / num_games) - avg_score^2
    # Calculate the standard deviation
    std_dev = sqrt(variance)
    avg_rss = player_rss .= player_rss ./ num_games
    avg_rbis = player_rbis .= player_rbis ./ num_games

    # Return the average score, standard deviation, max score, and min score
    return avg_score, variance, std_dev, max_score, min_score, avg_rss, avg_rbis
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


#sim_scores, sim_avg = get_sim(lineup, numSims)


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
ludf.var = Vector{Float64}(undef, nrow(ludf))
ludf.stdev = Vector{Float64}(undef, nrow(ludf))
ludf.maxval = Vector{Float64}(undef, nrow(ludf))
ludf.minval = Vector{Float64}(undef, nrow(ludf))

ludf.RSp1 = Vector{Float64}(undef, nrow(ludf))
ludf.RSp2 = Vector{Float64}(undef, nrow(ludf))
ludf.RSp3 = Vector{Float64}(undef, nrow(ludf))
ludf.RSp4 = Vector{Float64}(undef, nrow(ludf))
ludf.RSp5 = Vector{Float64}(undef, nrow(ludf))
ludf.RSp6 = Vector{Float64}(undef, nrow(ludf))
ludf.RSp7 = Vector{Float64}(undef, nrow(ludf))
ludf.RSp8 = Vector{Float64}(undef, nrow(ludf))
ludf.RSp9 = Vector{Float64}(undef, nrow(ludf))

ludf.RBIp1 = Vector{Float64}(undef, nrow(ludf))
ludf.RBIp2 = Vector{Float64}(undef, nrow(ludf))
ludf.RBIp3 = Vector{Float64}(undef, nrow(ludf))
ludf.RBIp4 = Vector{Float64}(undef, nrow(ludf))
ludf.RBIp5 = Vector{Float64}(undef, nrow(ludf))
ludf.RBIp6 = Vector{Float64}(undef, nrow(ludf))
ludf.RBIp7 = Vector{Float64}(undef, nrow(ludf))
ludf.RBIp8 = Vector{Float64}(undef, nrow(ludf))
ludf.RBIp9 = Vector{Float64}(undef, nrow(ludf))

seenLineups = Dict{String, Tuple}()

println("Number of simulations per game: ", numSims)

for i in 1:nrow(ludf)
    row = ludf[i,:]
    lineup_str = row.Lineup

    global lineup
    global player
    global der
    lineup_str = strip(lineup_str, ['[', ']'])

    if haskey(seenLineups, lineup_str)
        # Use the cached value
        (avg, vari, stdev, maxx, minn, avg_rs, avg_rbi) = seenLineups[lineup_str]
        ludf.score2[i] = avg
        ludf.var[i] = vari
        ludf.stdev[i] = stdev
        ludf.maxval[i] = maxx
        ludf.minval[i] = minn
        ludf.RSp1[i] = avg_rs[1]
        ludf.RSp2[i] = avg_rs[2]
        ludf.RSp3[i] = avg_rs[3]
        ludf.RSp4[i] = avg_rs[4]
        ludf.RSp5[i] = avg_rs[5]
        ludf.RSp6[i] = avg_rs[6]
        ludf.RSp7[i] = avg_rs[7]
        ludf.RSp8[i] = avg_rs[8]
        ludf.RSp9[i] = avg_rs[9]
        ludf.RBIp1[i] = avg_rbi[1]
        ludf.RBIp2[i] = avg_rbi[2]
        ludf.RBIp3[i] = avg_rbi[3]
        ludf.RBIp4[i] = avg_rbi[4]
        ludf.RBIp5[i] = avg_rbi[5]
        ludf.RBIp6[i] = avg_rbi[6]
        ludf.RBIp7[i] = avg_rbi[7]
        ludf.RBIp8[i] = avg_rbi[8]
        ludf.RBIp9[i] = avg_rbi[9]
        
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
          
              println("Processing lineup: ", lineup_str)
              
              avg, vari, stdev, maxx, minn, avg_rs, avg_rbi = average_score(lineup, numSims)
              println(avg, " ", vari," ",  stdev," ",  maxx," ",  minn," ",  avg_rs," ",  avg_rbi)
              seenLineups[lineup_str] = (avg, vari, stdev, maxx, minn, avg_rs, avg_rbi)

              ludf.score2[i] = avg
              ludf.var[i] = vari
              ludf.stdev[i] = stdev
              ludf.maxval[i] = maxx
              ludf.minval[i] = minn
              ludf.RSp1[i] = avg_rs[1]
              ludf.RSp2[i] = avg_rs[2]
              ludf.RSp3[i] = avg_rs[3]
              ludf.RSp4[i] = avg_rs[4]
              ludf.RSp5[i] = avg_rs[5]
              ludf.RSp6[i] = avg_rs[6]
              ludf.RSp7[i] = avg_rs[7]
              ludf.RSp8[i] = avg_rs[8]
              ludf.RSp9[i] = avg_rs[9]
              ludf.RBIp1[i] = avg_rbi[1]
              ludf.RBIp2[i] = avg_rbi[2]
              ludf.RBIp3[i] = avg_rbi[3]
              ludf.RBIp4[i] = avg_rbi[4]
              ludf.RBIp5[i] = avg_rbi[5]
              ludf.RBIp6[i] = avg_rbi[6]
              ludf.RBIp7[i] = avg_rbi[7]
              ludf.RBIp8[i] = avg_rbi[8]
              ludf.RBIp9[i] = avg_rbi[9]
        end
    end
end
        

println(ludf.score2)
CSV.write("playersim.csv", ludf)

