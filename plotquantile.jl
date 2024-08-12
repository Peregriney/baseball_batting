ENV["PLOTS_TEST"]=true
ENV["GKSwstype"]=100

using CSV, DataFrames, Distributed, Base.Threads, Random, Plots

#Plot expected number probability by run (sum of r_i*P(r_i) for r_i in 1:R)

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


exp_rs = Float64[]

if length(ARGS) < 1
  println("maxR = 40 by default")
  lengthmax = 40
elseif length(ARGS) == 1
  lengthmax = parse(Int64, ARGS[1])
  println("custom specified maxR is ", lengthmax)
end

for lngth in 1:lengthmax
    # Compute expected number for the current length
    expected_number = sum(runidxs[i] * rprobs[i] for i in 1:lngth)
    println("r, e[r], expected runs ", lngth, ",  ", rprobs[lngth], ",  ", expected_number)
    push!(exp_rs, expected_number)
end

scatter(collect(1:lengthmax),exp_rs, xlabel="R",ylabel="Expected R (cumulative sum)")
savefig("quantile.png")

