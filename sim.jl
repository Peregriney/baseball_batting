using Random
using DataFrames, CSV
using StatsBase
using Permutations

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
    # Add runs for players on second and third base
    bases[4] += bases[2] + bases[3]
    # Clear second and third base
    bases[2] = 0
    bases[3] = 0
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

function try_read()
    try 
        csv_str = ARGS[1]
        data_csv = CSV.read(csv_str, DataFrame)
        return data_csv
    catch e
        println("Error: first argument must be CSV file, could not read. Format as [pathname.csv] to command-line. Defaulting to redsox_2023.csv")
        data_csv = CSV.read("redsox_2023.csv",DataFrame)
        return data_csv
    end
end
playersData = CSV.read("redsox_2023.csv", DataFrame)
# Read probabilities and check that each player's stats sum to 1
if length(ARGS) <1
    print("No filename specified, defaulting to redsox_2023.csv")
    playersData = CSV.read("redsox_2023.csv", DataFrame)
else
    playersData = try_read()
end

lineup = (1,2,3,4,5,6,7,8,9)
print("average, std, max, min")
avg, stdev, maxx, minn = average_score(lineup, 10000)



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


scores, avg = get_sim(lineup, 10000)


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

count_array = count_occurrences(scores)


function sorted_values_by_keys(dict::Dict{Int, Float64})
    # Get the sorted keys
    sorted_keys = sort(collect(keys(dict)))
    
    # Get the values sorted by their keys
    sorted_values = [dict[key]* 10000 for key in sorted_keys]
    
    return sorted_values
end

sorted_values = sorted_values_by_keys(probmemo)
n = length(count_array)

plot(bar(collect(0:n-1),[sorted_values[1:n] count_array ], label=["DP" "sim"]),xlabel="Score (# of runs)", ylabel="Frequency (# of games)", title="Histogram for DP-extrapolated vs. Simulation", legend=true)
savefig("histogram-comparison.png")

sliced_longer_array = sorted_values[1:n]
difference = sliced_longer_array .- count_array
# Plot the difference by index
plot(difference, label="Difference", xlabel="Index", ylabel="Difference (DP P(r)*10000 - Sim count)", title="Element-wise Difference of DP - Sim", xticks = 0:1:n)
savefig("dpsim-difference.png")
