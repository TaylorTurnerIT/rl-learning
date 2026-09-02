pico-8 cartridge
version 42
__lua__
function _init()
 cls(0)
 fillp(0b1010010110100101.1)
 rectfill(0,0,7,7,7)
 fillp()
 for y=0,7 do
  local row=""
  for x=0,7 do
   row=row..(x==0 and "" or ",")..tostr(pget(x,y))
  end
  printh("__fillp__"..tostr(y).."|"..row)
 end
 cls(0)
 fillp(0b101101001011010.1)
 rectfill(0,0,7,7,7)
 fillp()
 for y=0,7 do
  local row=""
  for x=0,7 do
   row=row..(x==0 and "" or ",")..tostr(pget(x,y))
  end
  printh("__fillp2__"..tostr(y).."|"..row)
 end
 exit()
end
__gfx__
