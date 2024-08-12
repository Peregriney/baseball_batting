using CSV, DataFrames, Distributed, Base.Threads, Random

# Define the state space
struct State
    b::Int
    j::Int
    outs::Int
    base1::Int
    base2::Int
    base3::Int
end

memo = Dict{State, Float64}()

# f performs memoized recursion for an inning's possible states
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

###NOTE: THIS example seemed to have a typo in the word doc. The case f(b,j,outs,0,1,0 = .... triple.
#Seems like it should actually match to f(b,j,outs,0,0,1) according to the actual logic of the transition, so that's how I implemented it.
    elseif base1 == 0 && base2 == 0 && base3 == 1
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
    #bases are loaded :o
    elseif base1 == 1 && base2 == 1 && base3 == 1
        value = f(State(b,j-1,outs-1,base1,base2,base3)) * OutGet(b_next) +
                f(State(b, j - 1, outs, 1, 1, 0)) * Walk(b_next) +
                f(State(b, j - 1, outs, 1, 0, 1)) * Walk(b_next) +
                f(State(b, j - 1, outs, 0, 1, 1)) * Walk(b_next) +
                f(State(b, j - 1, outs, 1, 1, 1)) * Walk(b_next) +
                f(State(b, j - 1, outs, 1, 1, 0)) * Single(b_next) +
                f(State(b, j - 1, outs, 1, 1, 1)) * Single(b_next)

    end

    # Store the value in the memo
    memo[state] = value

    # Return the value --> value is the PROBABILITY for a state that:
    #given first batter is Player(b), then after j batters have come to bat, there are OUTS outs and b1-b3 represent boolean presence of players on the bases
    return value
end

function runsScored(state::State)::Float64
    b, j, outs, base1, base2, base3 = state.b, state.j, state.outs, state.base1, state.base2, state.base3
    return j - 3 - base1 - base2 - base3
end


h_memo = Dict{Tuple{Int, Int, Int}, Float64}()
g_memo = Dict{Tuple{Int, Int, Int}, Float64}()

function g(b::Int, b_prime::Int, r::Int)::Float64
  #equals sum of all states in memo WHERE outs == 3 AND b=b and j = b_prime and r = runsScored(state)
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

function g_thread(b::Int, b_prime::Int, r::Int)::Float64
    sumstore = Threads.Atomic{Float64}(0.0)
    @threads for (state, value) in collect(memo)

        if state.outs == 3 && state.b == b && Batters(state.b, state.j) == b_prime && r == runsScored(state)
            Threads.atomic_add!(sumstore, value)
        end
    end
    return sumstore[]
end

function h(i::Int, b_prime::Int, r::Int)::Float64
  # the probability that the last batter in inning i_cur is b, and runs (summing also the innings < i_cur) = r
    if i == 1
        return g(Batter(1), b_prime, r)
    end

    #consider cases i > 1 where we must sum previous innings too
    sum = 0.0
    for b in 1:9
        for r_prime in 0:r ####WAIT!! Do I sum to r or to 40? plz double-check.

            sum += h(i-1, b, r_prime) * g(Next(b),b_prime, r-r_prime)
        end
    end
    return sum
end

function h_parallel(i::Int, b_prime::Int, r::Int)::Float64
    key = (i, b_prime, r)
    if haskey(h_memo, key)
        return h_memo[key]
    end

    if i == 1
        result = g(Batter(1), b_prime, r)
    else
        result = 0.0
        for b in 1:9
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


#exclusively for test, implies i=1 only
function h_t(i::Int, b_prime::Int, r::Int)::Float64
    return g_thread(b_prime, b_prime, r)
end

#populate all entries in memo: b can be 1:9, j can be 1:27, outs can be 0:3, b1,b2,b3 can all be 0/1
function populateMemo()
    for b in 1:9
        #println(b, " b for populateMemo")
        for j in 1:27 ###Change Parameters after test
            for outs in -1:3
                for b1 in 0:1
                    for b2 in 0:1
                        for b3 in 0:1
                            f(State(b,j,outs,b1,b2,b3))
                        end
                    end
                end
            end
        end
    end
end

probmemo = Dict{Int, Float64}()
function prob(r::Int, INGS::Int)::Float64

    sum = 0.0
    #println("entering the prob function, 1:9")
    for b in 1:9
        sum += h_parallel(INGS,b,r)
    end
    println("p(", r, ") is ", sum)
    probmemo[r] = sum
    return sum
end


function expectedRuns(rmax::Int, INGS::Int)::Float64
    sum = Atomic{Float64}(0.0)  # Use an atomic variable for thread-safe summation

    @threads for r in 0:rmax
        #println(r, " expected runs r value")
        Threads.atomic_add!(sum, r * prob(r, INGS))  # Safely add to the atomic sum
    end

    return sum[]
end

# Define the functions for player statistics --> all of these need to retrieve from some array of values for each of the 9players
function Single(i::Int)::Float64
    # CSV structure ref. based on simulator.ipynb code
    return playersData[i,5]
end

function Double(i::Int)::Float64
    # CSV structure ref. based on simulator.ipynb code
    return playersData[i,6]
end

function Triple(i::Int)::Float64
    # CSV structure ref. based on simulator.ipynb code
    return playersData[i,7]
end

function HomeRun(i::Int)::Float64
    # CSV structure ref. based on simulator.ipynb code, function based on simulator.ipynb too
    playersInfo = playersData[i, 1:8]
    # Extract the probabilities for each outcome for the player
    playerSO = playersInfo[2] # Strikeout probability
    playerOO = playersInfo[3] # Other out probability
    playerBB = playersInfo[4] # Walk probability
    player1B = playersInfo[5] # Single probability
    player2B = playersInfo[6] # Double probability
    player3B = playersInfo[7] # Triple probability
    playerHR = 1 - (playerSO + playerOO + playerBB + player1B + player2B + player3B) # Home run probability
    return playerHR
end

function Walk(i::Int)::Float64
    # CSV structure ref. based on simulator.ipynb code
    return playersData[i,4]
end

function OutGet(i::Int)::Float64
    # CSV structure ref. based on simulator.ipynb code
    return playersData[i,2] + playersData[i,3]
end

# Define the function for the next batter
function Next(b::Int)::Int
    # --> checked, this is correct
    return (b % 9) + 1
end

# Define the function for the j-th batter in an inning
function Batters(b::Int, j::Int)::Int
    # Replace this with the actual logic for determining the j-th batter in an inning --> I haven't checked this actually, I think it's right but not 100% sure.
    return (b + j - 1) % 9 + 1
end

## Function to check if the probabilities in each row sum to 1
#taken from simulator.ipynb :)
function check_probabilities(playersData)
    for i in 1:nrow(playersData)
        row = playersData[i, :]
        probability_sum = sum(row[2:8])  # Adjust the indices as per your data structure
        if probability_sum < 0.99 || probability_sum > 1.01  # Checking if sum ~= 1
            println("Error: Probabilities in row ", i, " do not sum to 1. Sum is: ", probability_sum)
        else
            println("Row ", i, ": ", "Good")
        end
    end
    println("checked probs")
end

function try_read()
    try 
        csv_str = ARGS[1]
        data_csv = CSV.read(csv_str, DataFrame)
    catch e
        println("Error: first argument must be CSV file, could not read. Format as [pathname.csv] to command-line. Defaulting to redsox_2023.csv")
        data_csv = CSV.read("redsox_2023.csv",DataFrame)
    end
    return data_csv
end
playersData = CSV.read("redsox_2023.csv", DataFrame)
# Read probabilities and check that each player's stats sum to 1
if length(ARGS) <1
    print("No filename specified, defaulting to redsox_2023.csv")
    playersData = CSV.read("redsox_2023.csv", DataFrame)
else
    playersData = try_read()
end


check_probabilities(playersData)


function parse_args(args)
    try
        return [parse(Int64, arg) for arg in args]
    catch e
        println("Error: All arguments must be numbers. Setting random lineup.")
        lineup = randperm(9)
        println("lineup", lineup)
    end
end

# Currently, batting order (lineup) is just randomly initialized
if length(ARGS) <2 || length(ARGS) != 10
    println("batting lineup initialized to random. No lineup given") 
    lineup = randperm(9)
    println("lineup", lineup)


elseif length(ARGS) == 10
    lineup = parse_args(ARGS[2:10])
    println("accepted batting lineup", lineup)
end

function Batter(idx::Int)::Int
    return lineup[idx]
end

# Begin main code -- populate memo, etc. 
println("populating memo")
@time populateMemo()
println("calculating expected runs")

@time e = expectedRuns(40,9)
println(e)

# Convert the probmemo dictionary to a DataFrame
df = DataFrame(r = keys(probmemo), probability = values(probmemo))
# Save the DataFrame to a CSV file
CSV.write("probmemo.csv", df)


hi_values = Int[]
hbprime_values = Int[]
hr_values = Int[]
h_values = Float64[]
for (key, value) in h_memo
    push!(hi_values, key[1])
    push!(hbprime_values, key[2])
    push!(hr_values, key[3])
    push!(h_values,value)
end

gb_values = Int[]
gbprime_values = Int[]
gr_values = Int[]
g_values = Float64[]
for (key, value) in g_memo
    push!(gb_values, key[1])
    push!(gbprime_values, key[2])
    push!(gr_values, key[3])
    push!(g_values,value)
end

h_df = DataFrame(i = hi_values,
                    bprime = hbprime_values,
                    r = hr_values,
                    value = h_values)
g_df = DataFrame(b = gb_values,
                     bprime = gbprime_values,
                     r = gr_values,
                     value = g_values)

# Save h_memo DataFrame to CSV
CSV.write("hmemo.csv", h_df)

# Save g_memo DataFrame to CSV
CSV.write("gmemo.csv", g_df)


# Initialize arrays to store the data
b_values = Float64[]
j_values = Float64[]
outs_values = Int[]
base1_values = Int[]
base2_values = Int[]
base3_values = Int[]
value_values = Float64[]

# Iterate through keys and values of the dictionary once
for (key, value) in memo
    push!(b_values, key.b)
    push!(j_values, key.j)
    push!(outs_values, key.outs)
    push!(base1_values, key.base1)
    push!(base2_values, key.base2)
    push!(base3_values, key.base3)
    push!(value_values, value)
end

# Create the DataFrame from the collected data
memo_df = DataFrame(b = b_values,
                     j = j_values,
                     outs = outs_values,
                     base1 = base1_values,
                     base2 = base2_values,
                     base3 = base3_values,
                     value = value_values)

# Save the DataFrame to a CSV file
CSV.write("fmemo.csv", memo_df)

#= 
#Single-Inning Test Cases
@time e = expectedRuns(1, 1)
println(e)

@time e2 = expectedRuns(5, 1)
println(e2)
@time e3 = expectedRuns(10, 1)
println(e3)
@time e4 = expectedRuns(40, 1)
println(e4)
=#
