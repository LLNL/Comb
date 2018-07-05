
function read_test(input)
   local test = {}

   for i=1,#input do
      local line = input[i]

      local start, stop = string.find(input[i], "Starting ")
      if (start == 1) then
         -- first line is the test name
         assert(test.name == nil)
         test.name = line
      else
         -- read test data
         local start, stop, part_name, part_num, part_sum, part_min, part_max = string.find(input[i], "^([^%s]-): num (%d+) sum ([^%s]+) s min ([^%s]+) s max ([^%s]+) s")

         local part = {
               ["name"]=part_name,
               ["num"]=tonumber(part_num),
               ["sum"]=tonumber(part_sum),
               ["min"]=tonumber(part_min),
               ["max"]=tonumber(part_max)
            }
         test[#test+1] = part
         assert(test[part.name] == nil)
         test[part.name] = part
      end

   end

   return test
end

function merge_test(inout, input)

   if (inout.name == nil) then
      inout.name = input.name
   else
      assert(inout.name == input.name)
   end

   for i=1,#input do
      local part = input[i]

      if (inout[part.name] == nil) then
         inout[part.name] = {
               ["name"]=part.name,
               ["num"]=part.num,
               ["sum"]=part.sum,
               ["min"]=part.min,
               ["max"]=part.max
            }
         inout[#inout+1] = inout[part.name]
      else
         merge_part = inout[part.name]
         merge_part.num = merge_part.num + part.num
         merge_part.sum = merge_part.sum + part.sum
         merge_part.min = (merge_part.min < part.min) and merge_part.min or part.min
         merge_part.max = (merge_part.max > part.max) and merge_part.max or part.max
      end
   end
end

function print_test(f, input)

   f:write(string.format("%s\n", input.name))

   for _,part in ipairs(input) do

      -- pre-comm: num 100 sum 0.086043444 s min 0.000814067 s max 0.000900054 s
      f:write(string.format("%s: num %d sum %.9f s min %.9f s max %.9f s\n",
            part.name, part.num, part.sum, part.min, part.max))
   end
end


--Sample Input File
--[[
Started rank 62 of 64
Node rzmanta21
GPU 0 visible 0
OMP num threads 4
OMP thread map 0 1 2 3
Do mock communication
Cart coords    3    3    2
Message policy cutoff 200
Post Recv using wait_any method
Post Send using wait_any method
Wait Recv using wait_any method
Wait Send using wait_all method
Num cycles  100
Num cycles  100
Num vars    3
ghost_width 1
size           400      400      400
divisions        4        4        4
periodic         1        1        1
division map
map              0        0        0
map            100      100      100
map            200      200      200
map            300      300      300
map            400      400      400
Starting up memory pools
Host: num 1 sum 0.003041537 s min 0.003041537 s max 0.003041537 s
HostPinned: num 1 sum 0.014320877 s min 0.014320877 s max 0.014320877 s
Device: num 1 sum 0.003123093 s min 0.003123093 s max 0.003123093 s
Managed: num 1 sum 0.004590882 s min 0.004590882 s max 0.004590882 s
ManagedHostPreferred: num 1 sum 0.005733022 s min 0.005733022 s max 0.005733022 s
ManagedHostPreferredDeviceAccessed: num 1 sum 0.006578081 s min 0.006578081 s max 0.006578081 s
ManagedDevicePreferred: num 1 sum 0.001580223 s min 0.001580223 s max 0.001580223 s
ManagedDevicePreferredHostAccessed: num 1 sum 0.001520492 s min 0.001520492 s max 0.001520492 s
Starting test Mesh seq Host Buffers seq Host seq Host
pre-comm: num 100 sum 0.086043444 s min 0.000814067 s max 0.000900054 s
post-recv: num 100 sum 0.000727506 s min 0.000005992 s max 0.000009386 s
post-send: num 100 sum 0.045347361 s min 0.000435475 s max 0.000475411 s
wait-recv: num 100 sum 0.028533154 s min 0.000241260 s max 0.000439488 s
wait-send: num 100 sum 0.000441180 s min 0.000003514 s max 0.000015125 s
post-comm: num 100 sum 0.098815506 s min 0.000886734 s max 0.001017785 s
start-up: num 1 sum 0.001954850 s min 0.001954850 s max 0.001954850 s
test-comm: num 1 sum 0.164792819 s min 0.164792819 s max 0.164792819 s
bench-comm: num 1 sum 0.270929696 s min 0.270929696 s max 0.270929696 s
Starting test Mesh omp Host Buffers seq Host seq Host
pre-comm: num 100 sum 0.162776307 s min 0.001610004 s max 0.001658586 s
post-recv: num 100 sum 0.000491009 s min 0.000004293 s max 0.000005662 s
post-send: num 100 sum 0.059617311 s min 0.000571813 s max 0.000613337 s
wait-recv: num 100 sum 0.027558252 s min 0.000263427 s max 0.000298162 s
wait-send: num 100 sum 0.000415538 s min 0.000003279 s max 0.000010849 s
post-comm: num 100 sum 0.172885904 s min 0.001706967 s max 0.001780780 s
start-up: num 1 sum 0.000732745 s min 0.000732745 s max 0.000732745 s
test-comm: num 1 sum 0.023256641 s min 0.023256641 s max 0.023256641 s
bench-comm: num 1 sum 0.482504243 s min 0.482504243 s max 0.482504243 s
]]

function do_read(infiles, outfiles)

   for _, infile in ipairs(infiles) do
      infile.handle = assert(io.open(infile.name, "r"))

      local outfile_name = string.gsub(infile.name, "^(.-)_%d+$", "%1_combined")
      if (outfiles[outfile_name] == nil) then
         outfiles[outfile_name] = {
               ["name"]=outfile_name,
               ["header"]={},
               ["tests"]={}
            }
         outfiles[#outfiles+1] = outfiles[outfile_name]
      end

      infile.outfile = outfiles[outfile_name]
      infile.header = {}
      infile.tests = {}

      local context = infile.header
      local num_lines = 0
      while (true) do

         local line = infile.handle:read()

         if line == nil then break end

         num_lines = num_lines + 1

         if (string.find(line, "Started ") == 1) then
            -- write into header portion
            context = infile.header
         elseif (string.find(line, "Starting ") == 1) then
            -- write new test into tests portion
            assert(infile.tests[line] == nil)
            infile.tests[line] = {["name"]=line}
            infile.tests[#infile.tests+1] = infile.tests[line]
            context = infile.tests[line]
         end

         context[#context+1] = line

      end

      infile.handle:close()

   end

end

function do_combining(infiles, outfiles)

   for i=1,#infiles do
      local infile = infiles[i]

      local outfile = infile.outfile

      -- handle header

      if (outfile.header == nil) then
         outfile.header = {}
      end
      for j=1,#infile.header do
         local line = infile.header[j]

         local start, stop, rank, nprocs = string.find(line, "Started rank (%d+) of (%d+)")
         if (start == 1) then
            line = string.format("Started %d ranks", nprocs)
         end

         local start, stop = string.find(line, "Node ")
         if (start == 1) then
            line = string.format("Node")
         end

         local start, stop = string.find(line, "GPU ")
         if (start == 1) then
            line = string.format("GPU")
         end

         local start, stop = string.find(line, "OMP thread map ")
         if (start == 1) then
            line = string.format("OMP thread map")
         end

         local start, stop = string.find(line, "Cart coords ")
         if (start == 1) then
            line = string.format("Cart coords")
         end

         if (outfile.header[j] == nil) then
            outfile.header[j] = line
         else
            assert(outfile.header[j] == line)
         end
      end

      -- handle tests portion

      for j=1,#infile.tests do
         local test = infile.tests[j]
         local name = test.name

         if (outfile.tests[name] == nil) then
            outfile.tests[name] = {["name"]=name}
            outfile.tests[#outfile.tests+1] = outfile.tests[name]
         end
         merge_test(outfile.tests[name], read_test(test))

      end

   end

end

function do_output(infiles, outfiles)

   for i=1,#outfiles do
      local outfile = outfiles[i]

      outfile.handle = assert(io.open(outfile.name, "w"))

      -- handle header

      for j=1,#outfile.header do
         outfile.handle:write(string.format("%s\n", outfile.header[j]))
      end

      -- handle tests portion

      for j=1,#outfile.tests do
         print_test(outfile.handle, outfile.tests[j])
      end

      outfile.handle:close()

   end

end

-- main
do

   local infiles = {}
   local outfiles = {}

   local i = 1
   while i <= #arg do
      if arg[i] == "--help" then -- setup first
         print("Give file names of form *{num} will write files of form * with combined output")
         os.exit()
      else -- add file to list
         infiles[#infiles+1] = {["name"]=arg[i]}
         i = i + 1
      end
   end

   do_read(infiles, outfiles)

   do_combining(infiles, outfiles)

   do_output(infiles, outfiles)

end
