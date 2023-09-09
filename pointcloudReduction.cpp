#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <thread>
#include <cmath>
#include <bitset>
#include <vector>
#include <chrono>

//Hey there, this program is an octree-based pointcloud downsampler.
//I set this up so that the desired level of most detail can be easily set using the constant below, with any number from 1-10, representing the number of layers beyond the root of the
//octree. At the highest precision, 10, a 60x60x60 meter overall scan would be reduced to 5x5x5cm chunks which each contain 8 points within them, a pretty decent level of precision for
//this application, which roughly halves the number of points in the cloud compared to the input. This removal process is uniform and done in such a way as to preserve as much information
//as possible. I've included a benchmarker to make it easier to evaluate performance, the vast majority of this program's runtime is reading and writing csvs, not the actual algorithm. 
//I've outlined the guiding ideas of this, but the main parts worth looking at is the octreegeneration function and the "coding" functions. This approach is, I assume, somewhat slower than
//a voxel-grid distribution, but it has better clarity and control, in my opinion. It's also a lot easier to work with this data, assuming you don't immediately push it all to as csv of
//course, compared to grid-based distribution. Thanks for your consideration!

const int LOD = 9;

//Could just skip the struct and make a 2d array, but this is more readable
struct point {
    float coords[3];
    
    point(){
        coords[0] = 0.0;
        coords[1] = 0.0;
        coords[2] = 0.0;
    }
};

//Octree Node. If any of its children are == nullptr, then it is a leaf, and thus will instead store points. 
struct OTNode {
    int code;
    OTNode* children[8];
    OTNode* parent;
    point points[8];
    short pointCount;

    OTNode() {
        code = 0;
        std::memset(points, 0, sizeof(point) * 8);
        for (int i = 0; i < 8; i++) {
            children[i] = nullptr;
        }
        parent = nullptr;
        pointCount = 0;
    }
};

//Get row size
int getLines(std::ifstream& inp) {
    int rows = 0;
    inp.clear();
    std::string line;
    inp.seekg(0, std::ios_base::beg);
    while (std::getline(inp, line)) {
        rows++;
    }
    return rows;
}

//Take csv input and add to pointcloud array
void populateCloud(std::ifstream& inp, point* cloud, int rows) {
    inp.clear();
    inp.seekg(6, std::ios_base::beg);
    int i = 0;
    std::string line;
    while (std::getline(inp, line)) {
        std::stringstream ss(line);
        int j = 0;
        while (std::getline(ss, line, ',')) {
            cloud[i].coords[j] = std::stof(line);
            j++;
        }
        i++;
    }
}

//get upper and lower bounds for our data. Helps with reanchoring and is needed for coding calculations
void getUpperBound(point* cloud, int n, point& out) {
    point max;
    for (int i = 0; i < 3; i++) {
        max.coords[i] = cloud[i].coords[i];
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            max.coords[j] = std::max(max.coords[j], cloud[i].coords[j]);
        }
    }
    out = max;
}

//get upper and lower bounds for our data. Helps with reanchoring and is needed for coding calculations
void getLowerBound(point* cloud, int n, point& out) {
    point min;
    for (int i = 0; i < 3; i++) {
        min.coords[i] = cloud[i].coords[i];
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            min.coords[j] = std::min(min.coords[j], cloud[i].coords[j]);
        }
    }
    out = min;
}

//Makes a lot of operations related to partitioning easier. We'll add these values back at the output step.
void zero(point* cloud, point lower, int n) {
    for (int i = 0; i < n; i++) {
        cloud[i].coords[0] += -1.0 * (lower.coords[0]);
        cloud[i].coords[1] += -1.0 * (lower.coords[1]);
        cloud[i].coords[2] += -1.0 * (lower.coords[2]);
    }
}

//undo the zeroing
void reanchor(point* cloud, point lower, int n) {
    for (int i = 0; i < n; i++) {
        cloud[i].coords[0] += (lower.coords[0]);
        cloud[i].coords[1] += (lower.coords[1]);
        cloud[i].coords[2] += (lower.coords[2]);
    }
}

//Helper function for encoding. I think there is a mathematical formula for this but I can't think of it on the spot.
int pos(float n) {
    return n > 0 ? 1 : 0;
}

//Quick offset countsort implementation for radixsort. Only compares the bit at bit. Rest is just basic countsort
void countSort(__int64* codes, int n, int bit) {
    __int64* output = new __int64[n];
    int cnts[2] = { 0 };
    for (int i = 0; i < n; i++) {
        cnts[1] += ((codes[i] >> bit) % 2);
    }
    cnts[1] = n - cnts[1];

    for (int i = 0; i < n; i++) {
        output[cnts[(codes[i] >> bit) % 2]++] = codes[i];
    }

    for (int i = 0; i < n; i++) {
        codes[i] = output[i];
    }
}

//Radixsort, just a container for series of countsorts. 
void radixSort(__int64* codes, int n) {
    for (int i = 0; i < 3 * LOD; i++) {
        countSort(codes, n, 32 + i);
    }
}

//Binary search for first instance of code in codes. Needed for Octree Construction
int getFirstInd(int code, __int64* codes, int bitOffset, int n, int high, int low) {
    if (high >= low) {
        int m = low + (high - low) / 2;
        if (m == 0 || code > (codes[m - 1] >> bitOffset) && (codes[m] >> bitOffset) == code) {
            return m;
        }
        else if (code > codes[m] >> bitOffset) {
            return getFirstInd(code, codes, bitOffset, n, high, m + 1);
        }
        else {
            return getFirstInd(code, codes, bitOffset, n, m-1, low);
        }
    }
    else {
        return -1;
    }
}

//Binary search for last instance of code in codes. Needed for Octree Construction
int getLastInd(int code, __int64* codes, int bitOffset, int n, int high, int low) {
    if (high >= low) {
        int m = low + (high - low) / 2;
        if (m == n-1 || code < (codes[m + 1] >> bitOffset) && (codes[m] >> bitOffset) == code) {
            return m;
        }
        else if (code < codes[m] >> bitOffset) {
            return getLastInd(code, codes, bitOffset, n, m - 1, low);
        }
        else {
            return getLastInd(code, codes, bitOffset, n, high, m+1);
        }
    }
    else {
        return -1;
    }
}

//Generate our tree codes. the upper 32 bits encode the position in the octree the point belongs, and the lower 32 bits the index in cloud of that point
//Note that this encoding scheme is the reason why we cannot use LOD values about 10. A major revision that I did not consider to be worth it would be to store codes as their own 64 bit
//int, and associate the index via some sort of collection. I felt that was less clean and readable, and realistically LOD levels above 10 probably aren't useful for a downsampler
void genCodes(__int64* out, point* cloud, float sideLen, int n) {
    for (int i = 0; i < n; i++) { 
        for (int j = 0; j < LOD; j++) {
            out[i] <<= 3; //shift bits of out to the left to make room for next level
            out[i] += pos(fmodf(cloud[i].coords[0], (sideLen / pow(2.0, j))) - (sideLen / pow(2.0, j + 1))) * 4; //adds 4 or 0
            out[i] += pos(fmodf(cloud[i].coords[1], (sideLen / pow(2.0, j))) - (sideLen / pow(2.0, j + 1))) * 2; //adds 2 or 0
            out[i] += pos(fmodf(cloud[i].coords[2], (sideLen / pow(2.0, j))) - (sideLen / pow(2.0, j + 1))); //adds 1 or 0
        }
        out[i] <<= 32; //shift our code bits to the top 32, then add position in cloud
        out[i] += i;
    }
}

//Octree Generation
//
void generateOctree(OTNode *root, __int64* codes, int n, point* cloud) {
    std::vector<OTNode*> front; //The front is the layer of the octree we are currently generating. The futurefront is the next iteration's front.
    std::vector<OTNode*> futureFront;

    front.push_back(root); //Push the root node to our front

    int depth = 0;

    while (depth <= LOD) {
        //Bit offset to be used for search functions. subtract the current depth from the highest level of detail bit for offset
        int offset = (32 + LOD*3) - (depth * 3);

        for (int i = 0; i < front.size(); i++) {

            //Select a node
            OTNode* curr = front.at(i);
            int start = getFirstInd(curr->code, codes, offset, n, n, 0);

            if (start == -1) {//If the node's location code doesn't match the location code for any points in the cloud, it is empty and we can skip it.
                continue;
            }

            //If there are more than 8 points that are within the node's bounds, and we haven't bottomed out our level of detail yet, give the node children and pass them to the next front
            if ((getLastInd(curr->code, codes, offset, n, n, start) - start > 8) && depth != LOD) {
                for (int j = 0; j < 8; j++) {
                    OTNode* node = new OTNode();
                    node->code = (curr->code << 3) + j;
                    curr->children[j] = node;
                    futureFront.push_back(node);
                }
            }
            else {//If we have bottomed out our LOD, or there are less than 8 points within the node, add points to the node instead of children
                int end = getLastInd(curr->code, codes, offset, n, n, start);
                if (end - start > 8) {
                    //This looks a little ugly, but all it's doing is ensuring that if we have more than 8 points within the node, an even spread of them will be stored,
                    //not just the first 8. This produces far more usaful data.
                    int tmp = 0;
                    for (float j = start; j < end; j += (float(end) - float(start)) / 8.0) {
                        curr->points[tmp] = cloud[__int32(codes[(int)(j)])]; //get our index from the cloud back from lower 32 of code, and set node points to it.
                        curr->pointCount++;
                        tmp++;
                    }
                }
                else {
                    end = (end - start);//If there are 8 or less points within, just add them all to the node.
                    for (int j = start; j < start + end; j++) {
                        curr->points[j - start] = cloud[__int32(codes[j])]; //get our index from the cloud back from lower 32 of code, and set node points to it.
                        curr->pointCount++;
                    }
                }
            }
        }
        front = futureFront;//Swap over the new front
        futureFront.clear();
        depth++;
    }
}

//Write our output csv
void writeBack(OTNode* curr, std::ofstream& out) {
    if (curr->children[0] != nullptr) {
        for (int i = 0; i < 8; i++) {
            writeBack(curr->children[i], out);
        }
    }
    else {
        for (int i = 0; i < curr->pointCount; i++) {
            out << curr->points[i].coords[0] << "," << curr->points[i].coords[1] << "," << curr->points[i].coords[2] << std::endl;
        }
    }
}

//Recursively write to the output csv
void writeBack(OTNode* root) {
    std::ofstream out("output.csv");
    if (root->children[0] != nullptr) {
        for (int i = 0; i < 8; i++) {
            writeBack(root->children[i], out);
        }
    }
    out.close();
}

int main()
{
    std::ifstream inp("input.csv");//Get our input size, then take all that data and add it to the cloud variable
    int rows = getLines(inp) - 1;

    point* cloud = new point[rows];

    populateCloud(inp, cloud, rows);

    inp.close();

    auto start = std::chrono::high_resolution_clock::now();//Time algorithm performance
    //Get upper and lower bounds of the provided pointcloud, then find the greatest difference between their coordinates. This is our side length for the root of the octree
    //I considered trying a more parallel implementation of this algorithm, but ultimately decided it wasn't worth it, the overhead for thread dispatch is too high. Here's 
    //one spot it was, though.
    point upper;
    point lower;
    std::thread ma(getUpperBound, cloud, rows, std::ref(upper));
    std::thread mi(getLowerBound, cloud, rows, std::ref(lower));

    ma.join();
    mi.join();

    //The full side length of the octree should be whatever the largest difference between a upper and lower bound for an axis is.
    float sideLen = std::max(upper.coords[0] - lower.coords[0], std::max(upper.coords[1] - lower.coords[1], upper.coords[2] - lower.coords[2]));
    zero(cloud, lower, rows);

    //Codes store the locational code for a point, as well as the index of that point. The upper 32 bits are the location, the lower are the index.
    __int64* codes = new __int64[rows];
    std::memset(codes, 0, sizeof(__int64) * rows);
    
    //Generate codes for our octree and sort them for use in the main octree generation algorithm
    genCodes(codes, cloud, sideLen, rows);
    radixSort(codes, rows);
    
    //Make our root for the tree
    OTNode* root = new OTNode();
    
    //reset the position of the points now that we've finished everything we needed them zeroed for
    reanchor(cloud, lower, rows);

    //Build the Octree, see function for more detail
    generateOctree(root, codes, rows, cloud);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Ran at LOD " << LOD << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;//finished running

    //write output to a csv file
    writeBack(root);
}