ENV["PLOTS_TEST"]=true
ENV["GKSwstype"]=100
using CSV, DataFrames, Distributed, Base.Threads, Random, Plots
probmemo = CSV.read("probmemo.csv", DataFrame)

keys_str = probmemo.r[1]
values_str = probmemo.probability[1]
# Function to parse the list of values from a string
function parse_keys(values_str::String)::Vector{Int}
    # Remove brackets and split by comma
    values_str = strip(values_str, ['[', ']'])
    elements = split(values_str, ",")
    # Convert the split strings to integers
    return parse.(Int, elements)
end

function parse_values(values_str::String)::Vector{Float64}
    # Remove brackets and split by comma
    values_str = strip(values_str, ['[', ']'])
    elements = split(values_str, ",")
    # Convert the split strings to integers
    return parse.(Float64, elements)
end

r_keys = parse_keys(keys_str)
r_probs = parse_values(values_str)
probmemo = Dict(r_keys[i] => r_probs[i] for i in 1:length(r_keys))


runidxs = [key for key in keys(probmemo)]
rprobs = [val for val in values(probmemo)]

# Calculate the expected number (mean)
expected_number = sum(runidxs[i] * rprobs[i] for i in 1:length(runidxs))

# Calculate the variance and standard deviation
mean_sq = sum((runidxs[i] ^ 2) * rprobs[i] for i in 1:length(runidxs))
variance = mean_sq - expected_number^2
std_dev = sqrt(variance)

# Find minimum and maximum
min_val = minimum(runidxs)
max_val = maximum(runidxs)

# Print results
println("Expected Number (Mean): $expected_number")

r_keys = collect(keys(probmemo))
r_prob_vals = collect(values(probmemo))

# Create scatterplot
scatter(r_keys, r_prob_vals, xlabel="Runs in game", ylabel="Probability", title="PMF for # of runs scored")
savefig("pmf.png")
