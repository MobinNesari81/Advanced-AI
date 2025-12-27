import util
import submission
from mapUtil import createStanfordMap, locationFromTag, makeTag
from grader import extractPath, printPath

cityMap = createStanfordMap()
startLocation = locationFromTag(makeTag("landmark", "gates"), cityMap)
waypointTags = [makeTag("landmark", "bookstore"), makeTag("landmark", "memorial_church")]
endTag = makeTag("landmark", "oval")

ucs = util.UniformCostSearch(verbose=0)
ucs.solve(submission.WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap))
path = extractPath(startLocation, ucs)

printPath(path, waypointTags=waypointTags, cityMap=cityMap, outPath="path.json")
